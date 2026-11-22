package aggregate

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"testing"

	"kllm/cluster"
	"kllm/cluster/dataset"
	"kllm/cluster/record"
)

var req = cluster.Request{ID: "r"}

// ans builds a response whose every token carries confidence conf.
func ans(node string, conf float32, tokens ...int32) cluster.Response {
	c := make([]float32, len(tokens))
	for i := range c {
		c[i] = conf
	}
	return cluster.Response{Node: node, Tokens: tokens, Conf: c}
}

func seq(from, n int) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = int32(from + i)
	}
	return s
}

// splice is `base` up to position at, then a different continuation.
func splice(base []int32, at, from int) []int32 {
	return append(append([]int32(nil), base[:at]...), seq(from, len(base)-at)...)
}

func agg(t *testing.T, name string) cluster.Aggregator {
	t.Helper()
	a, err := New(name)
	if err != nil {
		t.Fatal(err)
	}
	if a.Name() != name {
		t.Fatalf("strategy %q calls itself %q", name, a.Name())
	}
	return a
}

// near allows for confidences being float32 on the wire.
func near(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestMajority(t *testing.T) {
	resps := []cluster.Response{
		ans("a", .9, 1, 2), ans("b", .9, 1, 2), ans("c", .9, 1, 2), ans("d", .9, 7, 7), {Node: "e", Err: "down"},
	}
	d := agg(t, "majority").Aggregate(req, resps)
	if !cluster.Equal(d.Tokens, []int32{1, 2}) || fmt.Sprint(d.Winners) != "[a b c]" {
		t.Fatalf("decision: %+v", d)
	}
	if !near(d.Share, 0.75) || !near(d.Margin, 0.5) || d.Abstained {
		t.Fatalf("share=%g margin=%g", d.Share, d.Margin)
	}
	if d := agg(t, "majority").Aggregate(req, []cluster.Response{{Node: "e", Err: "down"}}); !d.Abstained || d.Tokens != nil {
		t.Fatalf("no answers must abstain: %+v", d)
	}
}

// Every strategy must give the same verdict whatever order the responses
// arrive in, including on ties — otherwise results are not reproducible.
func TestDecisionsAreOrderIndependent(t *testing.T) {
	base := []cluster.Response{
		ans("a", .9, 1, 2, 3), ans("b", .5, 1, 2, 9), ans("c", .7, 4, 4, 4), ans("d", .6, 1, 2, 3), ans("e", .8, 4, 4, 4),
		ans("f", .4, 8, 8, 8),
	}
	for _, name := range Names() {
		want := agg(t, name).Aggregate(req, base)
		for shift := 1; shift < len(base); shift++ {
			rotated := append(append([]cluster.Response(nil), base[shift:]...), base[:shift]...)
			if got := agg(t, name).Aggregate(req, rotated); !cluster.Equal(got.Tokens, want.Tokens) {
				t.Errorf("%s: rotation %d changed the answer from %v to %v", name, shift, want.Tokens, got.Tokens)
			}
		}
	}
}

func TestQuorumAbstainsWithoutAMajorityOfAllNodes(t *testing.T) {
	q := agg(t, "quorum")
	strong := []cluster.Response{ans("a", .9, 1), ans("b", .9, 1), ans("c", .9, 1), ans("d", .9, 2), ans("e", .9, 3)}
	if d := q.Aggregate(req, strong); d.Abstained || !near(d.Share, 0.6) {
		t.Fatalf("3 of 5 is a quorum: %+v", d)
	}
	// Two of three answers agree, but two nodes are down: 2 of 5 is no quorum.
	weak := []cluster.Response{ans("a", .9, 1), ans("b", .9, 1), ans("c", .9, 2), {Node: "d", Err: "x"}, {Node: "e", Err: "x"}}
	if d := q.Aggregate(req, weak); !d.Abstained || !cluster.Equal(d.Tokens, []int32{1}) {
		t.Fatalf("2 of 5 must abstain (but still name its best guess): %+v", d)
	}
}

// The scenario that separates the families. Four honest nodes: two exactly
// right, two that diverged late in different directions. Three colluders
// agree on a fabrication. Counting whole answers, the colluders are the
// largest bloc; every strategy that credits partial agreement sees that
// four nodes back the true prefix.
func TestHonestNodesThatSplitLateStillOutvoteColluders(t *testing.T) {
	truth := seq(1, 16)
	lie := seq(900, 16)
	resps := []cluster.Response{
		ans("a", .9, truth...), ans("b", .9, truth...),
		ans("c", .9, splice(truth, 12, 100)...), ans("d", .9, splice(truth, 9, 200)...),
		ans("e", .9, lie...), ans("f", .9, lie...), ans("g", .9, lie...),
	}
	// Borda falls with majority: three identical ballots each put the lie
	// first, and that is worth as much as the honest nodes' split support.
	for _, name := range []string{"majority", "borda"} {
		if d := agg(t, name).Aggregate(req, resps); !cluster.Equal(d.Tokens, lie) {
			t.Fatalf("%s should fall for the largest bloc here, got %v", name, d.Tokens)
		}
	}
	for _, name := range []string{"token-vote", "mbr"} {
		d := agg(t, name).Aggregate(req, resps)
		if !cluster.Equal(d.Tokens, truth) || fmt.Sprint(d.Winners) != "[a b]" {
			t.Errorf("%s chose %v (winners %v), want the truth", name, d.Tokens, d.Winners)
		}
	}
	d := agg(t, "token-vote").Aggregate(req, resps)
	// Tightest contest on the path: 4 vs 3 at the first token, 2 vs 1 at the last split.
	if !near(d.Margin, 1.0/7) || !near(d.Share, 2.0/7) {
		t.Fatalf("token-vote share=%g margin=%g", d.Share, d.Margin)
	}
}

func TestTokenVoteHandlesDifferentLengths(t *testing.T) {
	// One node stopped early. The longer answer has the votes.
	resps := []cluster.Response{ans("a", .9, 1, 2, 3, 4), ans("b", .9, 1, 2, 3, 4), ans("c", .9, 1, 2)}
	d := agg(t, "token-vote").Aggregate(req, resps)
	if !cluster.Equal(d.Tokens, []int32{1, 2, 3, 4}) {
		t.Fatalf("chose %v", d.Tokens)
	}
	// ...and when the short answer has them, the walk stops there.
	resps = []cluster.Response{ans("a", .9, 1, 2), ans("b", .9, 1, 2), ans("c", .9, 1, 2, 3, 4)}
	d = agg(t, "token-vote").Aggregate(req, resps)
	if !cluster.Equal(d.Tokens, []int32{1, 2}) || fmt.Sprint(d.Winners) != "[a b]" {
		t.Fatalf("chose %v %v", d.Tokens, d.Winners)
	}
}

// Confidence weighting earns its keep on honest replicas: the minority
// that is right is confident, the majority that took the same low-margin
// wrong turn says so.
func TestConfidenceRescuesAConfidentMinority(t *testing.T) {
	resps := []cluster.Response{
		ans("a", .9, 1, 2), ans("b", .9, 1, 2),
		ans("c", .5, 1, 9), ans("d", .5, 1, 9), ans("e", .5, 1, 9),
	}
	if d := agg(t, "majority").Aggregate(req, resps); !cluster.Equal(d.Tokens, []int32{1, 9}) {
		t.Fatal("majority should pick the common wrong turn")
	}
	d := agg(t, "confidence").Aggregate(req, resps)
	if !cluster.Equal(d.Tokens, []int32{1, 2}) || !near(d.Share, 1.8/3.3) {
		t.Fatalf("confidence: %+v", d)
	}
}

// ...and is exactly as trustworthy as the nodes reporting it.
func TestSpoofedConfidence(t *testing.T) {
	resps := []cluster.Response{
		ans("a", .6, 1, 2), ans("b", .6, 1, 2), ans("c", .6, 1, 2),
		ans("x", .999, 7, 7), ans("y", .999, 7, 7),
	}
	lie := []int32{7, 7}
	if d := agg(t, "confidence").Aggregate(req, resps); !cluster.Equal(d.Tokens, lie) {
		t.Fatal("plain confidence weighting should be fooled by claimed certainty")
	}
	if d := agg(t, "max-confidence").Aggregate(req, resps); !cluster.Equal(d.Tokens, lie) {
		t.Fatal("max-confidence should be fooled by claimed certainty")
	}
	d := agg(t, "confidence-clipped").Aggregate(req, resps)
	if !cluster.Equal(d.Tokens, []int32{1, 2}) || !near(d.Share, 0.6) {
		t.Fatalf("clipping should cap a spoofed vote at an ordinary one: %+v", d)
	}
	// Clipping keeps part of the signal: when most nodes are confident, an
	// unsure pair still counts for less, which settles what a head count
	// would call a tie.
	tied := []cluster.Response{
		ans("a", .9, 1, 2), ans("b", .9, 1, 2), ans("c", .3, 1, 9), ans("d", .3, 1, 9), ans("e", .9, 5, 5),
	}
	if d := agg(t, "confidence-clipped").Aggregate(req, tied); !cluster.Equal(d.Tokens, []int32{1, 2}) {
		t.Fatalf("an unsure pair should lose to a confident pair: %+v", d)
	}
	// ...and gives up the rest. The ceiling is the median, so when the
	// unsure nodes are the majority a confident minority can no longer
	// overrule them — the price of not believing claimed certainty.
	outnumbered := []cluster.Response{
		ans("a", .9, 1, 2), ans("b", .9, 1, 2), ans("c", .3, 1, 9), ans("d", .3, 1, 9), ans("e", .3, 1, 9),
	}
	if d := agg(t, "confidence").Aggregate(req, outnumbered); !cluster.Equal(d.Tokens, []int32{1, 2}) {
		t.Fatalf("plain confidence lets the confident minority win: %+v", d)
	}
	if d := agg(t, "confidence-clipped").Aggregate(req, outnumbered); !cluster.Equal(d.Tokens, []int32{1, 9}) {
		t.Fatalf("clipped confidence reduces to a head count here: %+v", d)
	}
}

func TestUnreportedConfidenceGetsTheTypicalWeight(t *testing.T) {
	silent := cluster.Response{Node: "s", Tokens: []int32{5}}
	resps := []cluster.Response{ans("a", .8, 1), ans("b", .4, 2), silent}
	w := confWeights(resps)
	if !near(w["s"], 0.6) { // median of the reported 0.8 and 0.4
		t.Fatalf("silent node weight = %g", w["s"])
	}
	if w := confWeights([]cluster.Response{silent}); !near(w["s"], 0.5) {
		t.Fatalf("with nothing reported the prior is 0.5, got %g", w["s"])
	}
}

func TestSingleIsOneNodesAnswer(t *testing.T) {
	resps := []cluster.Response{{Node: "a", Err: "down"}, ans("b", .9, 1), ans("c", .9, 2), ans("d", .9, 3)}
	s := agg(t, "single")
	picked := map[string]int{}
	for i := 0; i < 300; i++ {
		d := s.Aggregate(cluster.Request{ID: fmt.Sprint("r", i)}, resps)
		if d.Abstained || len(d.Winners) != 1 || d.Winners[0] == "a" {
			t.Fatalf("decision: %+v", d)
		}
		picked[d.Winners[0]]++
	}
	if len(picked) != 3 {
		t.Fatalf("single should spread across the live nodes: %v", picked)
	}
	if d := s.Aggregate(req, resps[:1]); !d.Abstained {
		t.Fatal("only a dead node: must abstain")
	}
}

func TestRegistry(t *testing.T) {
	if _, err := New("no-such-strategy"); err == nil {
		t.Fatal("unknown strategy accepted")
	}
	for _, name := range Names() {
		d := agg(t, name).Aggregate(req, nil)
		if !d.Abstained {
			t.Errorf("%s answered with no responses: %+v", name, d)
		}
		one := agg(t, name).Aggregate(req, []cluster.Response{ans("a", .9, 4, 2)})
		if one.Abstained && name != "quorum" || !cluster.Equal(one.Tokens, []int32{4, 2}) {
			t.Errorf("%s on a single answer: %+v", name, one)
		}
	}
}

// accuracy replays a generated dataset through a strategy.
func accuracy(t *testing.T, recs []record.Record, name string) float64 {
	t.Helper()
	a := agg(t, name)
	right := 0
	for _, rec := range recs {
		if d := a.Aggregate(rec.Request, cluster.Blind(rec.Responses)); !d.Abstained && cluster.Equal(d.Tokens, rec.Reference) {
			right++
		}
	}
	return float64(right) / float64(len(recs))
}

func load(t *testing.T, attack string, n, f, requests int) []record.Record {
	t.Helper()
	sc, err := dataset.Standard(attack, n, f, requests, 7)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	w := record.NewWriter(&buf)
	if _, err := dataset.Generate(context.Background(), sc, w, dataset.Options{}); err != nil {
		t.Fatal(err)
	}
	w.Flush()
	var recs []record.Record
	record.Scan(&buf, func(r record.Record) error { recs = append(recs, r); return nil })
	return recs
}

// On data from the lab, not hand-built cases: redundancy has to beat no
// redundancy, and the claims made for each family have to hold in aggregate.
func TestStrategiesOnGeneratedData(t *testing.T) {
	normal := load(t, "normal", 7, 0, 600)
	single := accuracy(t, normal, "single")
	for _, name := range []string{"majority", "token-vote", "mbr", "borda", "confidence", "confidence-clipped"} {
		if got := accuracy(t, normal, name); got < single+0.05 {
			t.Errorf("normal: %s = %.3f does not beat a single node (%.3f)", name, got, single)
		}
	}

	// Three colluders of seven: the honest majority is only 4, and it splits.
	collude := load(t, "collude", 7, 3, 600)
	maj, tok := accuracy(t, collude, "majority"), accuracy(t, collude, "token-vote")
	if tok < maj+0.05 {
		t.Errorf("collude f=3: token-vote %.3f should clearly beat majority %.3f", tok, maj)
	}

	// Spoofed certainty matters once the liars' inflated votes can outweigh
	// the honest nodes that got it right: three of seven.
	spoof := load(t, "overconfident", 7, 3, 600)
	plain, clipped := accuracy(t, spoof, "confidence"), accuracy(t, spoof, "confidence-clipped")
	if clipped < plain+0.2 {
		t.Errorf("overconfident f=3: clipped %.3f should clearly beat plain confidence %.3f", clipped, plain)
	}
	if mc := accuracy(t, spoof, "max-confidence"); mc > 0.05 {
		t.Errorf("max-confidence under spoofing should be near zero, got %.3f", mc)
	}
}
