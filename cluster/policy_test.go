package cluster

import (
	"fmt"
	"testing"
)

func resp(node string, ms float64, tokens ...int32) Response {
	return Response{Node: node, LatencyMs: ms, Tokens: tokens}
}

func nodesOf(rs []Response) string {
	s := ""
	for _, r := range rs {
		s += r.Node
	}
	return s
}

func TestAgreement(t *testing.T) {
	share, margin, n := Agreement([]Response{
		resp("a", 1, 1, 2), resp("b", 1, 1, 2), resp("c", 1, 1, 2), resp("d", 1, 9), {Node: "e", Err: "x"},
	})
	if n != 4 || share != 0.75 || margin != 0.5 {
		t.Fatalf("share=%g margin=%g answered=%d", share, margin, n)
	}
	if share, margin, n := Agreement([]Response{{Node: "e", Err: "x"}}); share != 0 || margin != 0 || n != 0 {
		t.Fatal("no answers must mean no agreement")
	}
	if _, margin, _ := Agreement([]Response{resp("a", 1, 1), resp("b", 1, 2)}); margin != 0 {
		t.Fatalf("a tie has margin 0, got %g", margin)
	}
}

func TestOrderIsAPerRequestPermutation(t *testing.T) {
	ids := []string{"a", "b", "c", "d", "e"}
	p := Policy{}.Normalize(len(ids))
	first := map[string]int{}
	for i := 0; i < 500; i++ {
		req := Request{ID: fmt.Sprintf("r%d", i)}
		o := Order(p, req, ids, nil)
		if len(o) != len(ids) || nodesOf([]Response{{Node: o[0]}}) == "" {
			t.Fatal("not a permutation")
		}
		if again := Order(p, req, ids, nil); fmt.Sprint(again) != fmt.Sprint(o) {
			t.Fatal("order is not deterministic per request")
		}
		first[o[0]]++
	}
	for _, id := range ids {
		if first[id] < 50 { // expect ~100 each
			t.Fatalf("node %s is first on only %d/500 requests: %v", id, first[id], first)
		}
	}

	trust := map[string]float64{"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.9, "e": 0.2}
	p.Order = "trust"
	o := Order(p, Request{ID: "x"}, ids, func(id string) float64 { return trust[id] })
	if got := fmt.Sprint(o[2:]); got != "[c e a]" || (o[0] != "b" && o[0] != "d") {
		t.Fatalf("trust order = %v", o)
	}
}

func TestReplayAll(t *testing.T) {
	all := []Response{
		resp("a", 100, 1), resp("b", 150, 1), resp("c", 5000, 1), {Node: "d", Err: "crashed", LatencyMs: 3},
	}
	out := Policy{Mode: ModeAll, TimeoutMs: 1000}.Replay(Request{ID: "r"}, all, nil)
	if out.Queried != 4 || out.LatencyMs != 1000 || out.Rounds != 1 {
		t.Fatalf("outcome: %+v", out)
	}
	if len(Answered(out.Used)) != 2 {
		t.Fatal("the hung node must count as a timeout, not an answer")
	}
	if all[2].Err != "" {
		t.Fatal("Replay mutated its input")
	}
}

func TestReplayFirstK(t *testing.T) {
	all := []Response{
		resp("slow", 400, 1, 1, 1, 1), resp("b", 120, 1, 1, 1, 1),
		resp("fast", 2, 7, 7, 7, 7), {Node: "dead", Err: "crashed", LatencyMs: 1}, resp("c", 100, 1, 1, 1, 1),
	}
	out := Policy{Mode: ModeFirstK, K: 2, TimeoutMs: 1000}.Replay(Request{ID: "r"}, all, nil)
	// dead (1ms, error) and fast (2ms) arrive first; c (100ms) completes the quorum.
	if got := nodesOf(out.Used); got != "deadfastc" {
		t.Fatalf("used %q", got)
	}
	if out.LatencyMs != 100 || out.Queried != 5 {
		t.Fatalf("outcome: %+v", out)
	}
	// Bill: fast 4 + c 4, plus b cancelled at 100/120 and slow at 100/400 of 4 tokens.
	if want := 8 + 4*100.0/120 + 4*100.0/400; out.Tokens < want-1e-9 || out.Tokens > want+1e-9 {
		t.Fatalf("tokens billed %g, want %g", out.Tokens, want)
	}

	// Not enough answers for the quorum: wait for everyone.
	out = Policy{Mode: ModeFirstK, K: 5, TimeoutMs: 1000}.Replay(Request{ID: "r"}, all, nil)
	if len(out.Used) != 5 || out.LatencyMs != 400 {
		t.Fatalf("short quorum: %+v", out)
	}
}

func TestReplayAdaptive(t *testing.T) {
	agree := []Response{resp("a", 100, 1), resp("b", 110, 1), resp("c", 120, 1), resp("d", 130, 1), resp("e", 140, 1)}
	p := Policy{Mode: ModeAdaptive, Initial: 3, Step: 2, TimeoutMs: 1000}
	out := p.Replay(Request{ID: "r"}, agree, nil)
	if out.Queried != 3 || out.Rounds != 1 || len(out.Used) != 3 {
		t.Fatalf("unanimous first round should stop: %+v", out)
	}

	// Trust order asks a,b,c first; c dissents, so the policy escalates and
	// pays for a second round.
	split := []Response{resp("a", 100, 1), resp("b", 110, 1), resp("c", 120, 9), resp("d", 130, 1), resp("e", 140, 1)}
	trust := map[string]float64{"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
	p.Order = "trust"
	out = p.Replay(Request{ID: "r"}, split, func(id string) float64 { return trust[id] })
	if out.Queried != 5 || out.Rounds != 2 || out.LatencyMs != 120+140 {
		t.Fatalf("disagreement should escalate: %+v", out)
	}

	// A crashed node in the first round is not agreement: keep asking.
	crashed := []Response{resp("a", 100, 1), resp("b", 110, 1), {Node: "c", Err: "x", LatencyMs: 1}, resp("d", 130, 1), resp("e", 140, 1)}
	out = p.Replay(Request{ID: "r"}, crashed, func(id string) float64 { return trust[id] })
	if out.Queried != 5 || out.Rounds != 2 {
		t.Fatalf("two answers of an initial three must escalate: %+v", out)
	}
}

func TestPolicyString(t *testing.T) {
	for p, want := range map[Policy]string{
		{}:                                   "all",
		{Mode: ModeFirstK, K: 3}:             "first-3",
		{Mode: ModeAdaptive}:                 "adaptive-3+2",
		{Mode: ModeAdaptive, Order: "trust"}: "adaptive-3+2-trust",
		{Mode: ModeAdaptive, Initial: 2, Step: 1}: "adaptive-2+1",
	} {
		if got := p.Normalize(7).String(); got != want {
			t.Errorf("%+v = %q, want %q", p, got, want)
		}
	}
}
