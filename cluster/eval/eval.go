// Package eval measures the cluster: it replays recorded datasets through
// aggregation strategies and dispatch policies and reports accuracy,
// robustness, latency, efficiency, attack resilience and detection quality.
//
// Everything is computed by replay over wait-for-all recordings, so every
// strategy and policy is scored on exactly the same node responses, and a
// difference between two rows of a table is a difference between methods,
// not between runs.
package eval

import (
	"fmt"
	"math"

	"kllm/cluster"
	"kllm/cluster/aggregate"
	"kllm/cluster/detect"
	"kllm/cluster/record"
)

// Method is an aggregation strategy under a dispatch policy.
type Method struct {
	Aggregator string         `json:"aggregator"`
	Policy     cluster.Policy `json:"policy"`
}

// PolicyName is "all", "first-4", "adaptive-3+2", ...
func (m Method) PolicyName() string {
	if m.Policy.Mode == "" {
		return cluster.ModeAll
	}
	return m.Policy.String()
}

func (m Method) String() string {
	if p := m.PolicyName(); p != cluster.ModeAll {
		return m.Aggregator + "@" + p
	}
	return m.Aggregator
}

// Metrics is one method's score on one condition.
type Metrics struct {
	Attack      string `csv:"attack" json:"attack"`
	Adversaries int    `csv:"adversaries" json:"adversaries"`
	Nodes       int    `csv:"nodes" json:"nodes"`
	Method      string `csv:"method" json:"method"`
	Aggregator  string `csv:"aggregator" json:"aggregator"`
	Policy      string `csv:"policy" json:"policy"`
	Requests    int    `csv:"requests" json:"requests"`

	// Accuracy. An abstention is not a correct answer, so it counts
	// against ExactMatch; Coverage and Selective separate the two effects.
	ExactMatch float64 `csv:"exact_match" json:"exact_match"` // answers identical to the reference

	// CI95 is the half-width of a 95% interval on ExactMatch. Across
	// several seeds it is taken from the spread between them, not from the
	// pooled request count: requests within a run are not independent
	// draws (see ci95).
	CI95 float64 `csv:"em_ci95" json:"em_ci95"`

	// The lowest and highest ExactMatch of any single seed. Pooling hides
	// the difference between "80% everywhere" and "100% on four runs, 0%
	// on the fifth"; for strategies that learn, the second happens.
	SeedMin float64 `csv:"em_seed_min" json:"em_seed_min"`
	SeedMax float64 `csv:"em_seed_max" json:"em_seed_max"`

	TokenAcc  float64 `csv:"token_acc" json:"token_acc"` // mean fraction of the reference matched before diverging
	Coverage  float64 `csv:"coverage" json:"coverage"`   // requests answered rather than abstained
	Selective float64 `csv:"selective" json:"selective"` // exact match among the answered

	// Attack resilience: of the requests on which an adversary submitted
	// a tampered answer, the fraction where that answer was returned.
	AttackSuccess float64 `csv:"attack_success" json:"attack_success"`

	// Latency until the policy had the answers it needed, in ms.
	LatencyMean float64 `csv:"latency_mean_ms" json:"latency_mean_ms"`
	LatencyP50  float64 `csv:"latency_p50_ms" json:"latency_p50_ms"`
	LatencyP95  float64 `csv:"latency_p95_ms" json:"latency_p95_ms"`
	LatencyP99  float64 `csv:"latency_p99_ms" json:"latency_p99_ms"`

	// Efficiency: what an answer cost.
	NodeCalls        float64 `csv:"node_calls" json:"node_calls"`                 // nodes asked per request
	Tokens           float64 `csv:"tokens" json:"tokens"`                         // tokens generated per request
	TokensPerCorrect float64 `csv:"tokens_per_correct" json:"tokens_per_correct"` // tokens generated per correct answer
}

// score accumulates one method's outcomes; several datasets (seeds) of the
// same condition can be pooled into it.
type score struct {
	n, exact, answered, exactAnswered int
	prefix                            float64
	attacked, succeeded               int
	latency                           []float64
	calls, tokens                     float64
	perSeed                           []float64 // exact match of each replayed dataset
}

// Evaluate replays one dataset through a fresh instance of the method.
func Evaluate(recs []record.Record, m Method) (Metrics, error) {
	var s score
	if err := s.replay(recs, m); err != nil {
		return Metrics{}, err
	}
	return s.metrics(m), nil
}

func (s *score) replay(recs []record.Record, m Method) error {
	agg, err := aggregate.New(m.Aggregator)
	if err != nil {
		return err
	}
	var trust func(string) float64
	if t, ok := agg.(interface{ Trust(string) float64 }); ok {
		trust = t.Trust
	}
	n0, exact0 := s.n, s.exact
	defer func() { s.perSeed = append(s.perSeed, ratio(float64(s.exact-exact0), float64(s.n-n0))) }()
	for _, rec := range recs {
		out := m.Policy.Replay(rec.Request, rec.Responses, trust)
		d := agg.Aggregate(rec.Request, cluster.Blind(out.Used))

		s.n++
		s.latency = append(s.latency, out.LatencyMs)
		s.calls += float64(out.Queried)
		s.tokens += out.Tokens
		right := !d.Abstained && cluster.Equal(d.Tokens, rec.Reference)
		if !d.Abstained {
			s.answered++
			if len(rec.Reference) > 0 {
				s.prefix += float64(cluster.LCP(d.Tokens, rec.Reference)) / float64(len(rec.Reference))
			}
		}
		if right {
			s.exact++
			s.exactAnswered++
		}

		var tampered [][]int32
		for _, r := range rec.Responses {
			if r.Label != nil && r.Label.Attacked && r.OK() {
				tampered = append(tampered, r.Tokens)
			}
		}
		if len(tampered) > 0 {
			s.attacked++
			if !d.Abstained && !right {
				for _, t := range tampered {
					if cluster.Equal(d.Tokens, t) {
						s.succeeded++
						break
					}
				}
			}
		}
	}
	return nil
}

func ratio(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return a / b
}

func (s *score) metrics(m Method) Metrics {
	n := float64(s.n)
	em := ratio(float64(s.exact), n)
	out := Metrics{
		Method: m.String(), Aggregator: m.Aggregator, Policy: m.PolicyName(), Requests: s.n,
		ExactMatch: em, TokenAcc: ratio(s.prefix, n),
		Coverage: ratio(float64(s.answered), n), Selective: ratio(float64(s.exactAnswered), float64(s.answered)),
		AttackSuccess: ratio(float64(s.succeeded), float64(s.attacked)),
		LatencyMean:   cluster.Mean(s.latency),
		LatencyP50:    cluster.Percentile(s.latency, 0.50),
		LatencyP95:    cluster.Percentile(s.latency, 0.95),
		LatencyP99:    cluster.Percentile(s.latency, 0.99),
		NodeCalls:     ratio(s.calls, n), Tokens: ratio(s.tokens, n),
		TokensPerCorrect: ratio(s.tokens, float64(s.exact)),
	}
	out.CI95 = ci95(em, s.n, s.perSeed)
	if len(s.perSeed) > 0 {
		out.SeedMin, out.SeedMax = cluster.Percentile(s.perSeed, 0), cluster.Percentile(s.perSeed, 1)
	}
	return out
}

// ci95 is the half-width of a 95% interval on an exact-match rate em
// measured over n requests, given the rate of each run that was pooled.
//
// One run can only offer the binomial interval of its own count, which
// treats its requests as independent draws. They are not. A strategy that
// learns is right or wrong for a whole run at a time — captured on one
// seed, fine on the next — and each seed is a different model, workload
// and cluster besides. So with several runs the interval is a t interval
// on the per-run rates, which measures exactly that run-to-run spread. It
// is never reported narrower than the binomial one: a handful of runs can
// agree by luck, and the pooled count is still a floor on the noise.
func ci95(em float64, n int, perRun []float64) float64 {
	if n == 0 {
		return 0
	}
	half := 1.96 * math.Sqrt(em*(1-em)/float64(n)) // normal approximation; fine at these sample sizes
	if k := len(perRun); k > 1 {
		mean := cluster.Mean(perRun)
		var ss float64
		for _, x := range perRun {
			ss += (x - mean) * (x - mean)
		}
		stderr := math.Sqrt(ss / float64(k-1) / float64(k))
		half = math.Max(half, tCritical(k-1)*stderr)
	}
	return half
}

// tCritical is the two-sided 95% critical value of Student's t with df
// degrees of freedom.
func tCritical(df int) float64 {
	table := [...]float64{12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
		2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
		2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042}
	switch {
	case df < 1:
		return math.Inf(1)
	case df <= len(table):
		return table[df-1]
	}
	// Beyond the table, the usual expansion around the normal quantile:
	// within 0.0002 of the exact value from 31 degrees of freedom up, and
	// tending to 1.96 without a step.
	const z = 1.959964
	n := float64(df)
	return z + (z*z*z+z)/(4*n) + (5*math.Pow(z, 5)+16*z*z*z+3*z)/(96*n*n)
}

// Detection scores the detectors on one condition.
type Detection struct {
	Attack      string `csv:"attack" json:"attack"`
	Adversaries int    `csv:"adversaries" json:"adversaries"`
	Nodes       int    `csv:"nodes" json:"nodes"`
	Requests    int    `csv:"requests" json:"requests"`

	// Per response, stateless: does Inspect flag the tampered responses?
	RespPrecision float64 `csv:"resp_precision" json:"resp_precision"`
	RespRecall    float64 `csv:"resp_recall" json:"resp_recall"`
	RespF1        float64 `csv:"resp_f1" json:"resp_f1"`

	// Per response, with memory: the adaptive strategy's Suspects.
	SuspPrecision float64 `csv:"susp_precision" json:"susp_precision"`
	SuspRecall    float64 `csv:"susp_recall" json:"susp_recall"`
	SuspF1        float64 `csv:"susp_f1" json:"susp_f1"`

	// Per node, at the end of the run: Tracker suspects against the
	// nodes that actually tampered.
	NodePrecision float64 `csv:"node_precision" json:"node_precision"`
	NodeRecall    float64 `csv:"node_recall" json:"node_recall"`
	NodeF1        float64 `csv:"node_f1" json:"node_f1"`

	// Requests between a node's first tampered answer and its becoming a
	// suspect, averaged over the adversaries that were caught.
	TimeToDetect  float64 `csv:"time_to_detect" json:"time_to_detect"`
	FalseSuspects int     `csv:"false_suspects" json:"false_suspects"` // honest nodes under suspicion at the end
}

type confusion struct{ tp, fp, fn int }

func (c *confusion) add(flagged, truth bool) {
	switch {
	case flagged && truth:
		c.tp++
	case flagged:
		c.fp++
	case truth:
		c.fn++
	}
}

func (c confusion) prf() (p, r, f1 float64) {
	p, r = ratio(float64(c.tp), float64(c.tp+c.fp)), ratio(float64(c.tp), float64(c.tp+c.fn))
	return p, r, ratio(2*p*r, p+r)
}

type detScore struct {
	n                int
	resp, susp, node confusion
	ttd              float64
	caught, falsely  int
}

// Detect scores the detectors on one dataset.
func Detect(recs []record.Record) Detection {
	var s detScore
	s.replay(recs)
	return s.detection()
}

func (s *detScore) replay(recs []record.Record) {
	cfg := detect.Default()
	agg := aggregate.Adaptive()
	tracker := agg.(interface{ Tracker() *detect.Tracker }).Tracker()

	firstAttack := map[string]int{} // node -> 1-based request index of its first tampered answer
	nodes := map[string]bool{}
	for i, rec := range recs {
		blind := cluster.Blind(rec.Responses)
		flagged := map[string]bool{}
		for _, f := range cfg.Inspect(blind) {
			if detect.Adversarial(f.Reason) {
				flagged[f.Node] = true
			}
		}
		suspected := map[string]bool{}
		for _, id := range agg.Aggregate(rec.Request, blind).Suspects {
			suspected[id] = true
		}
		for _, r := range rec.Responses {
			truth := r.Label != nil && r.Label.Attacked
			nodes[r.Node] = true
			if truth && firstAttack[r.Node] == 0 {
				firstAttack[r.Node] = i + 1
			}
			s.resp.add(flagged[r.Node], truth)
			s.susp.add(suspected[r.Node] || (flagged[r.Node] && !r.OK()), truth)
		}
		s.n++
	}

	suspects := map[string]detect.Suspect{}
	for _, sus := range tracker.Suspects() {
		suspects[sus.Node] = sus
	}
	for id := range nodes {
		sus, flagged := suspects[id]
		truth := firstAttack[id] > 0
		s.node.add(flagged, truth)
		switch {
		case flagged && truth:
			s.caught++
			s.ttd += float64(max(0, sus.DetectedAt-firstAttack[id]))
		case flagged:
			s.falsely++
		}
	}
}

func (s detScore) detection() Detection {
	d := Detection{Requests: s.n, FalseSuspects: s.falsely, TimeToDetect: ratio(s.ttd, float64(s.caught))}
	d.RespPrecision, d.RespRecall, d.RespF1 = s.resp.prf()
	d.SuspPrecision, d.SuspRecall, d.SuspF1 = s.susp.prf()
	d.NodePrecision, d.NodeRecall, d.NodeF1 = s.node.prf()
	return d
}

// DefaultMethods is the comparison the standard sweep runs: every
// aggregation strategy under wait-for-all, then the dispatch policies with
// a representative stateless and adaptive strategy each.
func DefaultMethods(nodes int) []Method {
	var ms []Method
	for _, a := range []string{
		"single", "majority", "quorum", "token-vote", "mbr", "borda",
		"confidence", "confidence-clipped", "max-confidence", "reputation", "adaptive",
	} {
		ms = append(ms, Method{Aggregator: a})
	}
	k := nodes/2 + 1
	firstK := cluster.Policy{Mode: cluster.ModeFirstK, K: k}
	escal := cluster.Policy{Mode: cluster.ModeAdaptive, Initial: 3, Step: 2, Margin: 1, Order: "hash"}
	trusted := escal
	trusted.Order = "trust"
	for _, p := range []cluster.Policy{firstK, escal} {
		for _, a := range []string{"majority", "token-vote", "adaptive"} {
			ms = append(ms, Method{Aggregator: a, Policy: p})
		}
	}
	ms = append(ms, Method{Aggregator: "adaptive", Policy: trusted})
	for _, a := range Ablations {
		ms = append(ms, Method{Aggregator: a})
	}
	return ms
}

// Ablations are the adaptive strategy with one defence removed each, plus
// the variant whose learned weights are unbounded.
var Ablations = []string{"adaptive-no-screen", "adaptive-no-trust", "adaptive-no-blocs", "adaptive-unbounded", "adaptive-naive"}

func isAblation(name string) bool {
	for _, a := range Ablations {
		if a == name {
			return true
		}
	}
	return false
}

// Validate checks that every method names a real strategy.
func Validate(ms []Method) error {
	for _, m := range ms {
		if _, err := aggregate.New(m.Aggregator); err != nil {
			return fmt.Errorf("method %s: %w", m, err)
		}
	}
	return nil
}
