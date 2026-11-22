package node

import (
	"context"
	"fmt"
	"math"
	"time"

	"kllm/cluster"
)

// --- deterministic hashing -------------------------------------------------
//
// Everything random in the lab is a pure function of seeds and inputs, so a
// dataset is reproducible bit for bit from its scenario file.

func mix(x uint64) uint64 { // splitmix64 finaliser
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

func fold(h, v uint64) uint64 { return mix(h*0x100000001b3 ^ mix(v)) }

func unit(x uint64) float64 { return float64(x>>11) / (1 << 53) } // [0,1)

func hashString(s string) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(s); i++ {
		h = (h ^ uint64(s[i])) * 0x100000001b3
	}
	return mix(h)
}

// SeedFor derives a stable per-node seed from a node id.
func SeedFor(id string) uint64 { return hashString(id) }

// --- the synthetic model ---------------------------------------------------

// SimModel is a deterministic synthetic autoregressive language model. For
// any context it has a top-1 token, a runner-up, and a probability on each,
// all derived by hashing the context. Two properties make it a useful
// stand-in for a real LM in robustness experiments:
//
//   - It is autoregressive: the context hash absorbs every emitted token, so
//     one changed token changes everything after it, exactly like a real
//     decode that has left the reference trajectory.
//   - Margins vary. Most steps are confident; a fraction (HardFrac) are
//     close calls, and those are where imperfect replicas diverge.
//
// The greedy decode (Reference) is the ground truth.
type SimModel struct {
	Seed     uint64
	Vocab    int     // token ids are in [0, Vocab)
	HardFrac float64 // fraction of decode steps that are low-margin
}

// DefaultSimModel matches the tiny test checkpoints' vocabulary.
func DefaultSimModel(seed uint64) SimModel {
	return SimModel{Seed: seed, Vocab: 512, HardFrac: 0.1}
}

func (m SimModel) start(prompt []int32) uint64 {
	h := mix(m.Seed ^ 0x6b6c6c6d) // "kllm"
	for _, t := range prompt {
		h = fold(h, uint64(uint32(t)))
	}
	return h
}

// next returns the two leading candidates for context h and their
// probabilities.
func (m SimModel) next(h uint64) (t1, t2 int32, p1, p2 float32) {
	r := mix(h ^ 0x746f6b31)
	v := uint64(m.Vocab)
	t1 = int32(r % v)
	t2 = int32((uint64(t1) + 1 + (r>>32)%(v-1)) % v) // any token but t1
	u, w := unit(mix(r^1)), unit(mix(r^2))
	if u < m.HardFrac {
		p1 = float32(0.36 + 0.20*w) // close call
	} else {
		p1 = float32(0.80 + 0.195*w)
	}
	// The runner-up takes most of what is left, but never out-scores the
	// winner: at a close call it sits just below p1.
	p2 = min((1-p1)*0.85, p1*0.9)
	return
}

// Reference is the model's greedy decode: the ground-truth continuation.
func (m SimModel) Reference(prompt []int32, n int) (tokens []int32, conf []float32) {
	h := m.start(prompt)
	tokens = make([]int32, 0, n)
	conf = make([]float32, 0, n)
	for i := 0; i < n; i++ {
		t1, _, p1, _ := m.next(h)
		tokens = append(tokens, t1)
		conf = append(conf, p1)
		h = fold(h, uint64(uint32(t1)))
	}
	return
}

// flipWeight is how exposed a decode step is to perturbation: ~1 for the
// closest calls, ~0 for confident steps. It is the simulator's version of
// "quantization error flips the argmax only where the top logits are close".
func flipWeight(p1 float32) float64 {
	x := float64(1-p1) / 0.64
	return x * x
}

// --- one replica -----------------------------------------------------------

// Sim is one replica of a SimModel, as an imperfect serving node would run
// it. Noise is how lossy the replica is (0 reproduces the reference
// exactly); it stands in for quantization, numerics, or sampling drift. A
// replica errs by taking the runner-up token at a low-margin step, and then
// continues consistently from there.
//
// Corr sets how much of that error is common-mode. Replicas of one model
// tend to make the same mistakes — at the limit, identical quantized
// replicas make identical ones, and no amount of voting helps. With
// probability Corr a request's perturbation is drawn from a stream shared by
// every replica instead of this replica's private one.
//
// OffPath is a modelling assumption, and the one confidence-based
// aggregation depends on: once a decode has left the reference trajectory
// it is in less probable territory, so the probabilities it reports from
// then on are scaled by OffPath (0.9 by default). That makes a wrong answer
// measurably, but only modestly, less confident than a right one — earlier
// wrong turns more so than late ones. Set it to 1 to remove the signal and
// see what confidence weighting is worth without it.
type Sim struct {
	M        SimModel
	Noise    float64
	Corr     float64
	OffPath  float64
	NodeSeed uint64

	// Latency model: Overhead + tokens*TokenTime, scaled by lognormal
	// jitter, with an occasional heavy-tail stall.
	Overhead  time.Duration
	TokenTime time.Duration
	Jitter    float64 // lognormal sigma
	TailProb  float64 // probability a request stalls
	TailMult  float64 // how much slower a stalled request is
}

// NewSim returns a replica with the lab's default latency model: ~100 ms for
// a 16-token answer, 25% jitter, and a 2% chance of a 4x stall.
func NewSim(m SimModel, nodeSeed uint64, noise float64) *Sim {
	return &Sim{
		M: m, Noise: noise, NodeSeed: nodeSeed, OffPath: 0.9,
		Overhead: 20 * time.Millisecond, TokenTime: 5 * time.Millisecond,
		Jitter: 0.25, TailProb: 0.02, TailMult: 4,
	}
}

func (s *Sim) Model() string {
	return fmt.Sprintf("sim-%d/noise=%.3g", s.M.Seed, s.Noise)
}

func (s *Sim) Generate(ctx context.Context, req cluster.Request) (Output, error) {
	if err := ctx.Err(); err != nil {
		return Output{}, err
	}
	if s.M.Vocab < 2 {
		return Output{}, fmt.Errorf("sim: vocab must be at least 2")
	}
	if len(req.Prompt) == 0 || req.MaxNewTokens <= 0 {
		return Output{}, fmt.Errorf("sim: empty prompt or non-positive max_new_tokens")
	}
	h := s.M.start(req.Prompt)
	private := fold(mix(s.NodeSeed), h)
	stream := private
	if unit(mix(private^0xc0)) < s.Corr {
		stream = fold(mix(s.M.Seed^0xc0ffee), h) // shared by every replica
	}

	out := Output{
		Tokens: make([]int32, 0, req.MaxNewTokens),
		Conf:   make([]float32, 0, req.MaxNewTokens),
		Label:  cluster.Label{Fault: FaultNone},
	}
	drift := float32(1) // becomes OffPath once the decode leaves the reference trajectory
	for i := 0; i < req.MaxNewTokens; i++ {
		t1, t2, p1, p2 := s.M.next(h)
		tok, p := t1, p1*drift
		if unit(fold(stream, uint64(i))) < s.Noise*flipWeight(p1) {
			tok, p = t2, p2*drift
			if s.OffPath > 0 {
				drift = float32(s.OffPath)
			}
		}
		out.Tokens = append(out.Tokens, tok)
		out.Conf = append(out.Conf, p)
		h = fold(h, uint64(uint32(tok)))
	}
	out.Latency = s.latency(fold(private, hashString(req.ID)), len(out.Tokens))
	return out, nil
}

func (s *Sim) latency(h uint64, tokens int) time.Duration {
	base := float64(s.Overhead) + float64(tokens)*float64(s.TokenTime)
	// Box–Muller: one standard normal from two uniforms.
	u1, u2 := unit(mix(h^1)), unit(mix(h^2))
	z := math.Sqrt(-2*math.Log(1-u1)) * math.Cos(2*math.Pi*u2)
	d := base * math.Exp(s.Jitter*z)
	if unit(mix(h^3)) < s.TailProb {
		d *= s.TailMult
	}
	return time.Duration(d)
}
