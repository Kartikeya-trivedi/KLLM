package node

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"kllm/cluster"
)

// Fault kinds. A Fault is a kind plus modifiers; the named presets below
// are the combinations the experiments use.
const (
	FaultCrash     = "crash"     // fail-stop: the request errors
	FaultStraggler = "straggler" // honest answer, late
	FaultRandom    = "random"    // uncoordinated Byzantine: random tokens
	FaultPerturb   = "perturb"   // subtle corruption: a few tokens substituted
	FaultCollude   = "collude"   // coordinated: every member emits the same wrong answer
	FaultTruncate  = "truncate"  // lazy: half an answer for half the work
)

// ErrCrashed is what an injected crash returns.
var ErrCrashed = errors.New("node crashed (injected fault)")

// fastLatency is how long a Fast adversary takes to answer.
const fastLatency = 2 * time.Millisecond

// Fault describes how a node misbehaves.
type Fault struct {
	Name string `json:"name,omitempty"` // dataset label; defaults to Kind
	Kind string `json:"kind"`

	// Rate is the fraction of requests attacked; 0 or 1 means all of them.
	// A colluding group derives the choice from its shared Secret and the
	// prompt, so its members strike on the same requests (a trigger, in
	// backdoor terms).
	Rate float64 `json:"rate,omitempty"`

	// After makes a sleeper: the node is honest for its first After
	// requests, building whatever reputation the aggregator tracks.
	After int `json:"after,omitempty"`

	// Keep is the fraction of the honest answer left intact. For collude
	// it is the length of the honest prefix (stealth: the closer to the
	// truth, the harder to tell apart by similarity); for perturb it is
	// the per-token survival probability.
	Keep float64 `json:"keep,omitempty"`

	// SpoofConf, if set, is reported as the confidence of every token —
	// the attack on confidence-weighted aggregation.
	SpoofConf float64 `json:"spoof_conf,omitempty"`

	// Secret identifies a colluding group.
	Secret uint64 `json:"secret,omitempty"`

	// DelayMs is added to every response (straggler, or a hang if large).
	DelayMs float64 `json:"delay_ms,omitempty"`

	// Fast makes attacked responses return almost immediately — the attack
	// on first-k dispatch, which trusts whoever answers first.
	Fast bool `json:"fast,omitempty"`
}

// Honest reports whether the profile leaves answers and timing untouched.
func (f Fault) Honest() bool {
	return (f.Kind == "" || f.Kind == FaultNone) && f.DelayMs == 0
}

func (f Fault) label() string {
	switch {
	case f.Name != "":
		return f.Name
	case f.Kind == "":
		return FaultNone
	}
	return f.Kind
}

var presets = map[string]Fault{
	"none":          {Kind: FaultNone},
	"crash":         {Kind: FaultCrash},
	"flaky":         {Kind: FaultCrash, Rate: 0.3},
	"straggler":     {Kind: FaultStraggler, DelayMs: 400},
	"hang":          {Kind: FaultStraggler, DelayMs: 60000},
	"random":        {Kind: FaultRandom},
	"perturb":       {Kind: FaultPerturb, Keep: 0.9},
	"truncate":      {Kind: FaultTruncate},
	"collude":       {Kind: FaultCollude, Secret: 1},
	"stealth":       {Kind: FaultCollude, Secret: 1, Keep: 0.75},
	"overconfident": {Kind: FaultCollude, Secret: 1, SpoofConf: 0.999},
	"intermittent":  {Kind: FaultCollude, Secret: 1, Rate: 0.3},
	"sleeper":       {Kind: FaultCollude, Secret: 1, After: 200},
	"fast":          {Kind: FaultCollude, Secret: 1, Fast: true},
}

// Preset returns a named fault profile.
func Preset(name string) (Fault, bool) {
	f, ok := presets[name]
	f.Name = name
	return f, ok
}

// PresetNames lists the built-in profiles.
func PresetNames() []string {
	names := make([]string, 0, len(presets))
	for n := range presets {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// ParseFault accepts a preset name or a JSON Fault object.
func ParseFault(s string) (Fault, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Fault{Kind: FaultNone, Name: FaultNone}, nil
	}
	if strings.HasPrefix(s, "{") {
		var f Fault
		if err := json.Unmarshal([]byte(s), &f); err != nil {
			return Fault{}, fmt.Errorf("fault: %w", err)
		}
		return f, f.validate()
	}
	if f, ok := Preset(s); ok {
		return f, nil
	}
	return Fault{}, fmt.Errorf("fault: unknown profile %q (have %s)", s, strings.Join(PresetNames(), ", "))
}

func (f Fault) validate() error {
	switch f.Kind {
	case "", FaultNone, FaultCrash, FaultStraggler, FaultRandom, FaultPerturb, FaultCollude, FaultTruncate:
	default:
		return fmt.Errorf("fault: unknown kind %q", f.Kind)
	}
	if f.Rate < 0 || f.Rate > 1 || f.Keep < 0 || f.Keep > 1 || f.SpoofConf < 0 || f.SpoofConf > 1 {
		return fmt.Errorf("fault: rate, keep and spoof_conf must be in [0,1]")
	}
	return nil
}

// Inject wraps a backend in a fault profile. seed is the node's private
// randomness; vocab bounds the tokens an adversary may emit.
func Inject(inner Backend, f Fault, seed uint64, vocab int) Backend {
	if f.Honest() {
		return inner
	}
	return &faulty{inner: inner, f: f, seed: mix(seed ^ 0xfa17), vocab: max(vocab, 2)}
}

type faulty struct {
	inner Backend
	f     Fault
	seed  uint64
	vocab int
	seen  atomic.Int64
}

func (b *faulty) Model() string { return b.inner.Model() }

func promptHash(prompt []int32) uint64 {
	h := uint64(0x70726f6d)
	for _, t := range prompt {
		h = fold(h, uint64(uint32(t)))
	}
	return h
}

// strikes decides whether request number n is attacked.
func (b *faulty) strikes(n int64, ph uint64, req cluster.Request) bool {
	switch b.f.Kind {
	case "", FaultNone, FaultStraggler:
		return false
	}
	if n <= int64(b.f.After) {
		return false
	}
	if b.f.Rate <= 0 || b.f.Rate >= 1 {
		return true
	}
	key := fold(b.seed, hashString(req.ID))
	if b.f.Kind == FaultCollude {
		key = mix(b.f.Secret ^ 0x7472) // the group picks its targets together
	}
	return unit(fold(key, ph)) < b.f.Rate
}

func (b *faulty) Generate(ctx context.Context, req cluster.Request) (Output, error) {
	n := b.seen.Add(1)
	ph := promptHash(req.Prompt)
	strike := b.strikes(n, ph, req)
	label := cluster.Label{Fault: b.f.label(), Attacked: strike}
	delay := time.Duration(b.f.DelayMs * float64(time.Millisecond))

	if strike && b.f.Kind == FaultCrash {
		return Output{Label: label}, ErrCrashed
	}

	// An adversary that fabricates the whole answer need not run the model.
	fabricated := b.f.Kind == FaultRandom || (b.f.Kind == FaultCollude && b.f.Keep == 0)
	var base Output
	if !(strike && b.f.Fast && fabricated) {
		var err error
		if base, err = b.inner.Generate(ctx, req); err != nil {
			base.Label = label
			return base, err
		}
	}
	out := base
	out.Label = label
	out.Latency += delay
	if !strike {
		return out, nil
	}

	length := len(base.Tokens)
	if length == 0 {
		length = req.MaxNewTokens
	}
	rnd := fold(fold(b.seed, ph), hashString(req.ID)) // private stream for this request
	tok := func(h uint64) int32 { return int32(h % uint64(b.vocab)) }
	differ := func(t, from int32) int32 { // nudge t so it is not `from`
		if t == from {
			return int32((int(t) + 1) % b.vocab)
		}
		return t
	}

	switch b.f.Kind {
	case FaultRandom:
		out.Tokens = make([]int32, length)
		for i := range out.Tokens {
			out.Tokens[i] = tok(fold(rnd, uint64(i)))
		}

	case FaultPerturb:
		keep := b.f.Keep
		if keep == 0 {
			keep = 0.9
		}
		out.Tokens = append([]int32(nil), base.Tokens...)
		forced := int(fold(rnd, 0xf0) % uint64(length)) // at least one token changes
		for i := range out.Tokens {
			if i == forced || unit(fold(rnd, uint64(i))) >= keep {
				out.Tokens[i] = differ(tok(fold(rnd, uint64(i)+1<<32)), base.Tokens[i])
			}
		}

	case FaultCollude:
		k := min(int(b.f.Keep*float64(length)), length-1)
		target := fold(mix(b.f.Secret^0x636f6c), ph) // same for every member of the group
		out.Tokens = make([]int32, length)
		for i := range out.Tokens {
			switch {
			case i < k:
				out.Tokens[i] = base.Tokens[i]
			case i == k && i < len(base.Tokens):
				out.Tokens[i] = differ(tok(fold(target, uint64(i))), base.Tokens[i])
			default:
				out.Tokens[i] = tok(fold(target, uint64(i)))
			}
		}

	case FaultTruncate:
		half := max(1, length/2)
		out.Tokens = append([]int32(nil), base.Tokens[:half]...)
		out.Latency = base.Latency/2 + delay
	}

	// Confidence: an adversary reports whatever it likes. By default it
	// wears an honest-looking profile; SpoofConf claims near-certainty.
	out.Conf = make([]float32, len(out.Tokens))
	for i := range out.Conf {
		switch {
		case b.f.SpoofConf > 0:
			out.Conf[i] = float32(b.f.SpoofConf)
		case i < len(base.Conf):
			out.Conf[i] = base.Conf[i]
		default:
			out.Conf[i] = 0.9
		}
	}
	if b.f.Fast {
		out.Latency = fastLatency + delay
	}
	return out, nil
}
