// Package dataset turns a scenario description into labelled data: it
// builds a cluster (honest replicas of varying quality plus whatever
// adversaries the scenario calls for), drives a workload through the
// orchestrator in wait-for-all mode, logs every node's answer next to the
// ground truth, and preprocesses the log into feature tables.
//
// Generation runs in virtual time and is a pure function of the scenario:
// the same scenario file yields the same dataset, byte for byte.
package dataset

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"net/http"
	"sort"
	"time"

	"kllm/cluster"
	"kllm/cluster/node"
	"kllm/cluster/orchestrator"
	"kllm/cluster/record"
)

// Attacks are the adversarial conditions of the standard suite, in the
// order reports list them. "normal" (no adversary) is the baseline.
var Attacks = []string{
	"crash", "straggler", "random", "perturb", "truncate",
	"collude", "stealth", "overconfident", "intermittent", "sleeper", "fast",
}

// Group is a set of identically configured nodes.
type Group struct {
	Count int     `json:"count"`
	Noise float64 `json:"noise"`          // replica lossiness (see node.Sim)
	Corr  float64 `json:"corr,omitempty"` // share of that error that is common-mode
	Slow  float64 `json:"slow,omitempty"` // latency multiplier; 0 means 1

	// The group's fault profile: a preset name, or an explicit Fault.
	// Neither means honest.
	Preset string      `json:"preset,omitempty"`
	Fault  *node.Fault `json:"fault,omitempty"`
}

func (g Group) fault() (node.Fault, error) {
	if g.Fault != nil {
		return *g.Fault, nil
	}
	return node.ParseFault(g.Preset)
}

// Scenario fully determines a dataset.
type Scenario struct {
	Name         string  `json:"name"`
	Seed         uint64  `json:"seed"`
	Requests     int     `json:"requests"`
	MaxNewTokens int     `json:"max_new_tokens"`
	PromptMin    int     `json:"prompt_min"`
	PromptMax    int     `json:"prompt_max"`
	Vocab        int     `json:"vocab"`
	HardFrac     float64 `json:"hard_frac"`
	TimeoutMs    float64 `json:"timeout_ms"`
	Groups       []Group `json:"groups"`
}

// WithDefaults fills unset fields with the lab's standard values.
func (sc Scenario) WithDefaults() Scenario {
	if sc.Requests <= 0 {
		sc.Requests = 500
	}
	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 16
	}
	if sc.PromptMin <= 0 {
		sc.PromptMin = 4
	}
	if sc.PromptMax < sc.PromptMin {
		sc.PromptMax = sc.PromptMin + 28
	}
	if sc.Vocab < 2 {
		sc.Vocab = 512
	}
	if sc.HardFrac <= 0 {
		sc.HardFrac = 0.1
	}
	if sc.TimeoutMs <= 0 {
		sc.TimeoutMs = 1000
	}
	return sc
}

// Model is the scenario's synthetic language model.
func (sc Scenario) Model() node.SimModel {
	return node.SimModel{Seed: sc.Seed, Vocab: sc.Vocab, HardFrac: sc.HardFrac}
}

// Nodes is the cluster size.
func (sc Scenario) Nodes() (n int) {
	for _, g := range sc.Groups {
		n += g.Count
	}
	return
}

// HonestNoise is the lossiness of the standard suite's honest replicas,
// cycled across them: a cluster of mixed quality (think fp32, 8-bit and
// 4-bit replicas), which is what makes weighting nodes worth studying.
var HonestNoise = []float64{0.05, 0.10, 0.20}

// Standard builds a scenario of the standard suite: n nodes of which f run
// the named attack profile and the rest are honest replicas of mixed
// quality. Adversaries run a noiseless replica — they know the right answer
// and deviate from it on purpose. attack "normal" (or f = 0) is the
// all-honest baseline.
func Standard(attack string, n, f, requests int, seed uint64) (Scenario, error) {
	if n < 1 || f < 0 || f > n {
		return Scenario{}, fmt.Errorf("dataset: need 0 <= adversaries (%d) <= nodes (%d)", f, n)
	}
	if attack == "normal" || attack == "" {
		attack, f = "normal", 0
	}
	sc := Scenario{Name: fmt.Sprintf("%s-f%d", attack, f), Seed: seed, Requests: requests}.WithDefaults()
	counts := make([]int, len(HonestNoise))
	for i := 0; i < n-f; i++ {
		counts[i%len(HonestNoise)]++
	}
	for i, c := range counts {
		if c > 0 {
			sc.Groups = append(sc.Groups, Group{Count: c, Noise: HonestNoise[i]})
		}
	}
	if f > 0 {
		fault, ok := node.Preset(attack)
		if !ok {
			return Scenario{}, fmt.Errorf("dataset: unknown attack %q", attack)
		}
		if fault.After > 0 {
			fault.After = sc.Requests / 2 // a sleeper wakes half-way through
		}
		sc.Groups = append(sc.Groups, Group{Count: f, Fault: &fault})
	}
	return sc, nil
}

// Workload is the scenario's request stream: seeded pseudo-random prompts.
func (sc Scenario) Workload() []cluster.Request {
	sc = sc.WithDefaults()
	rng := splitmix(sc.Seed ^ 0x776f726b)
	reqs := make([]cluster.Request, sc.Requests)
	for i := range reqs {
		p := make([]int32, sc.PromptMin+int(rng()%uint64(sc.PromptMax-sc.PromptMin+1)))
		for j := range p {
			p[j] = int32(rng() % uint64(sc.Vocab))
		}
		reqs[i] = cluster.Request{ID: fmt.Sprintf("%s-%06d", sc.Name, i), Prompt: p, MaxNewTokens: sc.MaxNewTokens}
	}
	return reqs
}

func splitmix(state uint64) func() uint64 {
	return func() uint64 {
		state += 0x9e3779b97f4a7c15
		z := state
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		return z ^ (z >> 31)
	}
}

// Lab is a scenario's cluster, built and ready to serve.
type Lab struct {
	Scenario Scenario
	Model    node.SimModel
	Nodes    []*node.Node

	// Faults is the answer key: node id -> fault profile ("none" if honest).
	Faults map[string]string
}

// Build instantiates the scenario's nodes. Ids are n00, n01, ... assigned
// in a seeded shuffle, so a node's name says nothing about its honesty.
func (sc Scenario) Build() (*Lab, error) {
	sc = sc.WithDefaults()
	type spec struct {
		g     Group
		fault node.Fault
	}
	var specs []spec
	for _, g := range sc.Groups {
		f, err := g.fault()
		if err != nil {
			return nil, fmt.Errorf("scenario %s: %w", sc.Name, err)
		}
		for i := 0; i < g.Count; i++ {
			specs = append(specs, spec{g, f})
		}
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("scenario %s: no nodes", sc.Name)
	}
	rng := splitmix(sc.Seed ^ 0x73687566)
	for i := len(specs) - 1; i > 0; i-- {
		j := int(rng() % uint64(i+1))
		specs[i], specs[j] = specs[j], specs[i]
	}

	lab := &Lab{Scenario: sc, Model: sc.Model(), Faults: map[string]string{}}
	for i, s := range specs {
		id := fmt.Sprintf("n%02d", i)
		seed := node.SeedFor(fmt.Sprintf("%s/%d/%s", sc.Name, sc.Seed, id))
		sim := node.NewSim(lab.Model, seed, s.g.Noise)
		sim.Corr = s.g.Corr
		if s.g.Slow > 0 {
			sim.Overhead = time.Duration(float64(sim.Overhead) * s.g.Slow)
			sim.TokenTime = time.Duration(float64(sim.TokenTime) * s.g.Slow)
		}
		backend := node.Inject(sim, s.fault, seed, sc.Vocab)
		lab.Nodes = append(lab.Nodes, node.New(node.Config{ID: id, Labels: true}, backend))
		label := node.FaultNone
		if !s.fault.Honest() {
			label = s.fault.Name
			if label == "" {
				label = s.fault.Kind
			}
		}
		lab.Faults[id] = label
	}
	return lab, nil
}

// Collect generates a scenario's dataset in memory.
func Collect(ctx context.Context, sc Scenario) ([]record.Record, *Lab, error) {
	var buf bytes.Buffer
	w := record.NewWriter(&buf)
	lab, err := Generate(ctx, sc, w, Options{})
	if err != nil {
		return nil, nil, err
	}
	if err := w.Flush(); err != nil {
		return nil, nil, err
	}
	var recs []record.Record
	err = record.Scan(&buf, func(r record.Record) error { recs = append(recs, r); return nil })
	return recs, lab, err
}

// Options tunes generation.
type Options struct {
	// HTTP puts every node behind a real loopback HTTP server instead of
	// calling it in-process: slower, but it exercises the whole wire path.
	HTTP bool

	// Metrics, if set, receives a system-metrics Sample every
	// MetricsEvery requests (default 100) and one at the end.
	Metrics      *record.Writer
	MetricsEvery int
}

// Generate runs the scenario's workload through its cluster and writes one
// Record per request to out, each carrying every node's labelled response
// and the ground-truth reference. It returns the lab (for its answer key).
func Generate(ctx context.Context, sc Scenario, out *record.Writer, opt Options) (*Lab, error) {
	lab, err := sc.Build()
	if err != nil {
		return nil, err
	}
	sc = lab.Scenario
	policy := cluster.Policy{Mode: cluster.ModeAll, TimeoutMs: sc.TimeoutMs}
	orch := orchestrator.New(orchestrator.Options{Policy: policy})
	for _, n := range lab.Nodes {
		if !opt.HTTP {
			orch.Add(n)
			continue
		}
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			return nil, err
		}
		srv := &http.Server{Handler: n.Handler()}
		go srv.Serve(ln)
		defer srv.Close()
		orch.Add(node.NewHTTPClient(n.ID(), "http://"+ln.Addr().String()))
	}
	every := opt.MetricsEvery
	if every <= 0 {
		every = 100
	}
	// A virtual clock: the run "starts" at a fixed instant and advances by
	// each request's latency, so timestamps are reproducible too.
	clock := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	sample := func(seq int) {
		if opt.Metrics != nil {
			opt.Metrics.Write(record.Sample{Time: clock, Scenario: sc.Name, Seq: seq, Nodes: orch.Stats(ctx)})
		}
	}

	for i, req := range sc.Workload() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		res, err := orch.Infer(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("scenario %s, request %d: %w", sc.Name, i, err)
		}
		sort.Slice(res.Responses, func(a, b int) bool { return res.Responses[a].Node < res.Responses[b].Node })
		rec := record.FromResult(res, i, clock)
		rec.Scenario = sc.Name
		rec.Reference, _ = lab.Model.Reference(req.Prompt, req.MaxNewTokens)
		rec.LatencyMs = policy.Replay(req, res.Responses, nil).LatencyMs
		if err := out.Write(rec); err != nil {
			return nil, err
		}
		clock = clock.Add(time.Duration(rec.LatencyMs * float64(time.Millisecond)))
		if (i+1)%every == 0 {
			sample(i + 1)
		}
	}
	if sc.Requests%every != 0 {
		sample(sc.Requests)
	}
	return lab, nil
}
