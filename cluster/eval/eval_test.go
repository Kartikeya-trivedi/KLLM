package eval

import (
	"context"
	"math"
	"strings"
	"testing"

	"kllm/cluster"
	"kllm/cluster/dataset"
	"kllm/cluster/record"
)

func load(t *testing.T, attack string, n, f, requests int) []record.Record {
	t.Helper()
	sc, err := dataset.Standard(attack, n, f, requests, 11)
	if err != nil {
		t.Fatal(err)
	}
	recs, _, err := dataset.Collect(context.Background(), sc)
	if err != nil {
		t.Fatal(err)
	}
	return recs
}

func evaluate(t *testing.T, recs []record.Record, m Method) Metrics {
	t.Helper()
	x, err := Evaluate(recs, m)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

// A hand-built log small enough to score by hand.
func TestMetricsOnAKnownLog(t *testing.T) {
	ref := []int32{1, 2, 3, 4}
	lie := []int32{9, 9, 9, 9}
	honest := func(node string, ms float64, tokens []int32) cluster.Response {
		return cluster.Response{Node: node, Tokens: tokens, LatencyMs: ms, Label: &cluster.Label{Fault: "none"}}
	}
	liar := func(node string) cluster.Response {
		return cluster.Response{Node: node, Tokens: lie, LatencyMs: 50, Label: &cluster.Label{Fault: "collude", Attacked: true}}
	}
	rec := func(resps ...cluster.Response) record.Record {
		return record.Record{Request: cluster.Request{ID: "r"}, Reference: ref, Responses: resps}
	}
	recs := []record.Record{
		// 1. Three right, one liar: correct. Attack present, failed.
		rec(honest("a", 100, ref), honest("b", 110, ref), honest("c", 120, ref), liar("x")),
		// 2. One right, two liars: the lie wins. Attack succeeded.
		rec(honest("a", 100, ref), liar("x"), liar("y")),
		// 3. No attack; the majority diverged at the third token: wrong, half the prefix right.
		rec(honest("a", 100, ref), honest("b", 100, []int32{1, 2, 7, 7}), honest("c", 300, []int32{1, 2, 7, 7})),
		// 4. No attack, unanimous.
		rec(honest("a", 100, ref), honest("b", 100, ref), honest("c", 200, ref)),
	}
	x := evaluate(t, recs, Method{Aggregator: "majority"})
	if x.Requests != 4 || x.ExactMatch != 0.5 || x.Coverage != 1 || x.Selective != 0.5 {
		t.Fatalf("accuracy: %+v", x)
	}
	if want := (1 + 0 + 0.5 + 1) / 4.0; x.TokenAcc != want {
		t.Fatalf("token accuracy %g, want %g", x.TokenAcc, want)
	}
	if x.AttackSuccess != 0.5 { // attacked on 2 requests, succeeded on 1
		t.Fatalf("attack success %g", x.AttackSuccess)
	}
	if x.NodeCalls != 3.25 || x.Tokens != 13 || x.TokensPerCorrect != 26 {
		t.Fatalf("efficiency: calls=%g tokens=%g per-correct=%g", x.NodeCalls, x.Tokens, x.TokensPerCorrect)
	}
	// Wait-for-all latency is the slowest node: 120, 100, 300, 200.
	if x.LatencyMean != 180 || x.LatencyP50 != 160 || x.Method != "majority" || x.Policy != "all" {
		t.Fatalf("latency: %+v", x)
	}
	if want := 1.96 * math.Sqrt(0.25/4); math.Abs(x.CI95-want) > 1e-12 {
		t.Fatalf("ci95 %g, want %g", x.CI95, want)
	}

	// A quorum abstains on the split request 2 (the lie has 2 of 3): coverage
	// drops, and the attack no longer succeeds.
	q := evaluate(t, []record.Record{recs[0], rec(honest("a", 100, ref), liar("x"), cluster.Response{Node: "y", Err: "down"})},
		Method{Aggregator: "quorum"})
	if q.Coverage != 0.5 || q.ExactMatch != 0.5 || q.Selective != 1 || q.AttackSuccess != 0 {
		t.Fatalf("quorum: %+v", q)
	}

	if _, err := Evaluate(recs, Method{Aggregator: "nope"}); err == nil {
		t.Fatal("unknown strategy accepted")
	}
}

func TestMethodNames(t *testing.T) {
	for m, want := range map[Method]string{
		{Aggregator: "majority"}: "majority",
		{Aggregator: "adaptive", Policy: cluster.Policy{Mode: cluster.ModeFirstK, K: 4}}:             "adaptive@first-4",
		{Aggregator: "mbr", Policy: cluster.Policy{Mode: cluster.ModeAdaptive, Initial: 3, Step: 2}}: "mbr@adaptive-3+2",
	} {
		if m.String() != want {
			t.Errorf("%+v = %q, want %q", m, m.String(), want)
		}
	}
	if err := Validate(DefaultMethods(7)); err != nil {
		t.Fatal(err)
	}
}

// Dispatch policies are a trade: the test pins the direction of each
// effect on generated data.
func TestPoliciesTradeReliabilityForCost(t *testing.T) {
	normal := load(t, "normal", 7, 0, 800)
	all := evaluate(t, normal, Method{Aggregator: "majority"})
	firstK := evaluate(t, normal, Method{Aggregator: "majority", Policy: cluster.Policy{Mode: cluster.ModeFirstK, K: 4}})
	escal := evaluate(t, normal, Method{Aggregator: "majority", Policy: cluster.Policy{Mode: cluster.ModeAdaptive, Initial: 3, Step: 2}})

	if firstK.LatencyP99 >= all.LatencyP99*0.6 {
		t.Errorf("first-4 p99 %.0f ms should be far below wait-for-all's %.0f ms", firstK.LatencyP99, all.LatencyP99)
	}
	if firstK.Tokens >= all.Tokens {
		t.Errorf("first-4 should bill fewer tokens (%.1f) than wait-for-all (%.1f)", firstK.Tokens, all.Tokens)
	}
	if escal.NodeCalls >= 6 || escal.NodeCalls < 3 {
		t.Errorf("escalation asked %.2f nodes on average; expected well under 7", escal.NodeCalls)
	}
	if escal.ExactMatch < all.ExactMatch-0.05 {
		t.Errorf("escalation lost too much accuracy: %.3f vs %.3f", escal.ExactMatch, all.ExactMatch)
	}
	if all.NodeCalls != 7 || all.Tokens != 7*16 {
		t.Errorf("wait-for-all cost: %+v", all)
	}

	// The price of first-k: whoever answers first decides.
	fast := load(t, "fast", 7, 2, 800)
	allFast := evaluate(t, fast, Method{Aggregator: "majority"})
	kFast := evaluate(t, fast, Method{Aggregator: "majority", Policy: cluster.Policy{Mode: cluster.ModeFirstK, K: 4}})
	if kFast.ExactMatch > allFast.ExactMatch-0.1 {
		t.Errorf("answer-first colluders should hurt first-4 (%.3f) far more than wait-for-all (%.3f)",
			kFast.ExactMatch, allFast.ExactMatch)
	}
}

func TestDetectionOnGeneratedData(t *testing.T) {
	clean := Detect(load(t, "normal", 7, 0, 600))
	if clean.FalseSuspects != 0 || clean.NodeRecall != 0 {
		t.Errorf("normal run: %+v", clean)
	}

	d := Detect(load(t, "collude", 7, 2, 600))
	if d.NodePrecision != 1 || d.NodeRecall != 1 || d.FalseSuspects != 0 {
		t.Errorf("collude f=2: node detection %+v", d)
	}
	if d.TimeToDetect > 15 {
		t.Errorf("collude f=2: took %.1f requests to catch a node that lies every time", d.TimeToDetect)
	}
	// The stateless detector misses a fabricated answer when the honest
	// answers on that request are themselves spread out (the robust
	// yardstick widens); memory closes the gap.
	if d.RespRecall < 0.8 || d.SuspRecall < 0.95 {
		t.Errorf("collude f=2: response recall %.3f, suspect recall %.3f", d.RespRecall, d.SuspRecall)
	}

	// Stealth is built to evade similarity: the stateless detector should
	// mostly miss it, and memory should mostly recover it.
	s := Detect(load(t, "stealth", 7, 2, 600))
	if s.RespRecall > 0.3 {
		t.Errorf("stealth f=2: the stateless detector caught %.3f — is stealth still stealthy?", s.RespRecall)
	}
	if s.SuspRecall < 0.8 || s.NodeRecall != 1 {
		t.Errorf("stealth f=2: with memory, suspect recall %.3f node recall %.3f", s.SuspRecall, s.NodeRecall)
	}
}

func TestSweepAndReport(t *testing.T) {
	s := Sweep{Nodes: 5, MaxAdversaries: 2, Requests: 150, Seeds: []uint64{1, 2}, Attacks: []string{"collude", "crash"},
		Methods: []Method{{Aggregator: "single"}, {Aggregator: "majority"}, {Aggregator: "adaptive"},
			{Aggregator: "majority", Policy: cluster.Policy{Mode: cluster.ModeFirstK, K: 3}}}}
	var seen int
	rep, err := s.Run(context.Background(), func(done, total int, _ string) { seen = max(seen, done) })
	if err != nil {
		t.Fatal(err)
	}
	if conds := 1 + 2*2; len(rep.Detection) != conds || len(rep.Metrics) != conds*4 || seen != conds {
		t.Fatalf("%d detection rows, %d metric rows, progress reached %d", len(rep.Detection), len(rep.Metrics), seen)
	}
	m, ok := rep.Get("collude", 2, "majority")
	if !ok || m.Requests != 300 || m.Nodes != 5 || m.Adversaries != 2 {
		t.Fatalf("pooled cell: %+v", m)
	}
	// Deterministic: the same sweep gives the same numbers.
	again, _ := s.Run(context.Background(), nil)
	if m2, _ := again.Get("collude", 2, "majority"); m2 != m {
		t.Fatalf("sweep is not reproducible:\n%+v\n%+v", m, m2)
	}

	rob := rep.Robustness()
	if len(rob) != 3 || rob[0].Method != "single" || len(rob[1].Worst) != 2 {
		t.Fatalf("robustness: %+v", rob)
	}
	maj := rob[1]
	if maj.Clean < 0.9 || maj.Worst[0] > maj.Clean+0.02 || maj.Worst[1] > maj.Worst[0]+0.02 {
		t.Fatalf("majority should not improve as adversaries are added: %+v", maj)
	}
	if rob[0].Clean >= maj.Clean {
		t.Fatalf("redundancy should beat a single node: %.3f vs %.3f", rob[0].Clean, maj.Clean)
	}

	md := rep.Markdown()
	for _, want := range []string{
		"## Exact match, 2 of 5 nodes adversarial", "| strategy | normal | collude | crash |",
		"## Robustness summary", "## Attack success rate", "## Dispatch policies", "majority@first-3", "## Detection",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("report is missing %q", want)
		}
	}

	if _, err := (Sweep{Methods: []Method{{Aggregator: "nope"}}}).Run(context.Background(), nil); err == nil {
		t.Fatal("a sweep with an unknown strategy ran")
	}
}
