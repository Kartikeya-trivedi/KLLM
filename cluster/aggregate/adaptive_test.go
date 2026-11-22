package aggregate

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"kllm/cluster"
	"kllm/cluster/dataset"
	"kllm/cluster/detect"
	"kllm/cluster/record"
)

type tracked interface {
	cluster.Aggregator
	Trust(string) float64
	Tracker() *detect.Tracker
}

// replay runs a dataset through one stateful aggregator instance, in
// order, and returns its accuracy over [from, to).
func replay(t *testing.T, a cluster.Aggregator, recs []record.Record, from, to int) float64 {
	t.Helper()
	right := 0
	for i, rec := range recs {
		d := a.Aggregate(rec.Request, cluster.Blind(rec.Responses))
		if i >= from && i < to && !d.Abstained && cluster.Equal(d.Tokens, rec.Reference) {
			right++
		}
	}
	return float64(right) / float64(to-from)
}

// faults reads the answer key out of a dataset's labels.
func faults(recs []record.Record) (honest, bad []string) {
	for _, r := range recs[0].Responses {
		if r.Label.Fault == "none" {
			honest = append(honest, r.Node)
		} else {
			bad = append(bad, r.Node)
		}
	}
	return
}

func TestAdaptiveStrategiesLearnWhoIsReliable(t *testing.T) {
	recs := load(t, "collude", 7, 2, 600)
	honest, bad := faults(recs)
	for _, name := range []string{"reputation", "adaptive"} {
		a := agg(t, name).(tracked)
		acc := replay(t, a, recs, 0, len(recs))
		if acc < 0.97 {
			t.Errorf("%s: accuracy %.3f with 2 colluders of 7", name, acc)
		}
		for _, id := range bad {
			if a.Trust(id) > 0.1 {
				t.Errorf("%s still trusts colluder %s at %.2f", name, id, a.Trust(id))
			}
		}
		for _, id := range honest {
			if a.Trust(id) < 0.6 {
				t.Errorf("%s distrusts honest node %s at %.2f", name, id, a.Trust(id))
			}
		}
		if got := fmt.Sprint(a.Tracker().Blocs()); got != fmt.Sprint([][]string{bad}) {
			t.Errorf("%s: blocs %s, want the two colluders %v", name, got, bad)
		}
	}
}

func TestAdaptiveReportsSuspects(t *testing.T) {
	recs := load(t, "collude", 7, 2, 300)
	_, bad := faults(recs)
	a := agg(t, "adaptive")
	var last cluster.Decision
	for _, rec := range recs {
		last = a.Aggregate(rec.Request, cluster.Blind(rec.Responses))
	}
	if fmt.Sprint(last.Suspects) != fmt.Sprint(bad) {
		t.Fatalf("suspects on the last request: %v, want %v", last.Suspects, bad)
	}
}

// The sleeper is the attack on learned trust: three nodes behave for the
// first half, earn a perfect record, then turn. Reputation weighting has
// been taught to prefer them, and pays for it until their record catches
// up; over the long run it recovers and ends ahead of plain majority.
// Adaptive gives a good record no more than an ordinary vote and reacts to
// the change itself, so it has no such window.
func TestSleeperAttack(t *testing.T) {
	recs := load(t, "sleeper", 7, 3, 800)
	half := len(recs) / 2
	names := []string{"majority", "token-vote", "reputation", "adaptive"}
	window, after := map[string]float64{}, map[string]float64{}
	for _, name := range names {
		window[name] = replay(t, agg(t, name), recs, half, half+40)
		after[name] = replay(t, agg(t, name), recs, half, len(recs))
	}
	t.Logf("first 40 requests after waking: %v", window)
	t.Logf("all %d requests after waking:  %v", len(recs)-half, after)

	if window["reputation"] > window["majority"]-0.1 {
		t.Errorf("just after the sleepers wake, reputation (%.3f) should trail plain majority (%.3f)",
			window["reputation"], window["majority"])
	}
	if window["adaptive"] < window["majority"] {
		t.Errorf("adaptive (%.3f) should have no such window (majority %.3f)", window["adaptive"], window["majority"])
	}
	if after["reputation"] < after["majority"] {
		t.Errorf("reputation (%.3f) should recover past majority (%.3f) in the long run", after["reputation"], after["majority"])
	}
	if after["adaptive"] < after["token-vote"]-0.01 || after["adaptive"] < after["reputation"] {
		t.Errorf("adaptive (%.3f) should match or beat token-vote (%.3f) and reputation (%.3f)",
			after["adaptive"], after["token-vote"], after["reputation"])
	}
}

// Strategies that learn from their own decisions are bistable near the
// breaking point: whichever side wins early is the side the strategy
// learns to trust. Stealth colluders at 3 of 7 win about a quarter of
// plain token votes — enough, on an unlucky run, to capture a strategy
// that learns from every decision. Learning only from decisive requests
// removes the captured state. (Seed 2 is such a run; the claim is checked
// on every seed.)
func TestLearningFromContestedDecisionsInvitesCapture(t *testing.T) {
	worst := map[string]float64{}
	var perSeed []string
	for seed := uint64(1); seed <= 4; seed++ {
		sc, err := dataset.Standard("stealth", 7, 3, 1000, seed)
		if err != nil {
			t.Fatal(err)
		}
		recs, _, err := dataset.Collect(context.Background(), sc)
		if err != nil {
			t.Fatal(err)
		}
		line := fmt.Sprintf("seed %d:", seed)
		for _, name := range []string{"token-vote", "reputation", "adaptive-naive", "adaptive"} {
			acc := replay(t, agg(t, name), recs, 0, len(recs))
			if w, ok := worst[name]; !ok || acc < w {
				worst[name] = acc
			}
			line += fmt.Sprintf(" %s=%.3f", name, acc)
		}
		perSeed = append(perSeed, line)
	}
	t.Log(strings.Join(perSeed, "\n"))

	if worst["reputation"] > 0.2 || worst["adaptive-naive"] > 0.2 {
		t.Errorf("expected reputation (worst %.3f) and naive adaptive (worst %.3f) to be captured on some seed",
			worst["reputation"], worst["adaptive-naive"])
	}
	if worst["adaptive"] < 0.95 {
		t.Errorf("adaptive should hold on every seed, worst was %.3f", worst["adaptive"])
	}
	if worst["token-vote"] > 0.85 || worst["token-vote"] < 0.6 {
		t.Errorf("token-vote was expected to sit near 0.75 here, worst %.3f", worst["token-vote"])
	}
}

// history plays 200 requests in which x, y, z agree and a..d each answer
// differently — three nodes out-voting four that split.
func history(a cluster.Aggregator) {
	right := seq(1, 16)
	for i := 0; i < 200; i++ {
		var resps []cluster.Response
		for _, id := range []string{"x", "y", "z"} {
			resps = append(resps, ans(id, .9, right...))
		}
		for j, id := range []string{"a", "b", "c", "d"} {
			resps = append(resps, ans(id, .9, splice(right, 4, 1000*(j+1)+i)...)) // each in its own way
		}
		a.Aggregate(cluster.Request{ID: fmt.Sprint(i)}, resps)
	}
}

// Three nodes that out-vote four split ones have won, but they have not
// proved anything: the default strategy does not learn from it.
func TestContestedDecisionsAreNotEvidence(t *testing.T) {
	a := agg(t, "adaptive").(tracked)
	history(a)
	for _, id := range []string{"a", "x"} {
		if got := a.Trust(id); got != 0.5 {
			t.Fatalf("trust in %s moved to %.2f on contested decisions alone", id, got)
		}
	}
	if len(a.Tracker().Suspects()) != 0 || a.Tracker().Observed() != 200 {
		t.Fatalf("suspects %v after %d requests", a.Tracker().Suspects(), a.Tracker().Observed())
	}
	naive := agg(t, "adaptive-naive").(tracked)
	history(naive)
	if naive.Trust("a") > 0.1 || naive.Trust("x") < 0.9 {
		t.Fatalf("the naive variant should have learnt it: a=%.2f x=%.2f", naive.Trust("a"), naive.Trust("x"))
	}
}

// Even with a bad history in hand, a learned discount must never overturn
// a head-count majority: that is what bounded influence guarantees.
func TestLearnedDiscountCannotOverturnAMajority(t *testing.T) {
	right, wrong := seq(1, 16), splice(seq(1, 16), 12, 500)
	// The four distrusted nodes now agree with each other against the three trusted ones.
	var resps []cluster.Response
	for _, id := range []string{"x", "y", "z"} {
		resps = append(resps, ans(id, .9, wrong...))
	}
	for _, id := range []string{"a", "b", "c", "d"} {
		resps = append(resps, ans(id, .9, right...))
	}

	bounded := AdaptiveWith(AdaptiveOptions{LearnAlways: true}).(tracked) // learns the bad history, but bounded
	history(bounded)
	if bounded.Trust("a") > 0.1 || bounded.Trust("x") < 0.9 {
		t.Fatalf("setup: trust a=%.2f x=%.2f", bounded.Trust("a"), bounded.Trust("x"))
	}
	if d := bounded.Aggregate(cluster.Request{ID: "final"}, resps); !cluster.Equal(d.Tokens, right) {
		t.Fatalf("four agreeing nodes were out-voted by three on reputation alone: winners %v", d.Winners)
	}

	naive := agg(t, "adaptive-naive")
	history(naive)
	if d := naive.Aggregate(cluster.Request{ID: "final"}, resps); !cluster.Equal(d.Tokens, wrong) {
		t.Fatalf("expected the unbounded variant to side with reputation, winners %v", d.Winners)
	}
}

func TestAdaptiveOnTheStandardAttacks(t *testing.T) {
	for _, c := range []struct {
		attack string
		f      int
		floor  float64
	}{
		{"normal", 0, 0.98}, {"random", 3, 0.95}, {"collude", 3, 0.93}, {"overconfident", 3, 0.93}, {"fast", 3, 0.93},
	} {
		recs := load(t, c.attack, 7, c.f, 600)
		got := replay(t, agg(t, "adaptive"), recs, 0, len(recs))
		maj := accuracy(t, recs, "majority")
		t.Logf("%-14s f=%d adaptive=%.3f majority=%.3f", c.attack, c.f, got, maj)
		if got < c.floor {
			t.Errorf("%s f=%d: adaptive accuracy %.3f below %.2f", c.attack, c.f, got, c.floor)
		}
	}
}
