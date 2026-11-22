package aggregate

import (
	"context"
	"fmt"
	"math"
	"math/rand"
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

// twinsHistory plays 200 requests in which a and b — honest replicas that
// share their mistakes — give the same wrong answer on every fourth one
// while everyone else is right. Each of those is settled by a 5-of-7
// majority, so the strategy learns from it: a and b end up with the worse
// record, and are flagged as a bloc.
func twinsHistory(a cluster.Aggregator) {
	right := seq(1, 16)
	for i := 0; i < 200; i++ {
		var resps []cluster.Response
		for _, id := range []string{"a", "b"} {
			tokens := right
			if i%4 == 0 {
				tokens = splice(right, 8, 7000+i)
			}
			resps = append(resps, ans(id, .9, tokens...))
		}
		for _, id := range []string{"c", "d", "x", "y", "z"} {
			resps = append(resps, ans(id, .9, right...))
		}
		a.Aggregate(cluster.Request{ID: fmt.Sprint(i)}, resps)
	}
}

// Sharing a vote is not bounded the way a discount is, which is why a
// majority has to be final: two honest replicas that share their mistakes
// look like a bloc, and one discounted vote between them would hand the
// answer of four agreeing nodes to three. The rule holds at every
// position: it covers the pair going wrong half-way through, and a lie
// that only departs from the truth half-way through.
func TestSharedVotesCannotOverturnAMajority(t *testing.T) {
	right := seq(1, 16)
	against := func(pair, lie []int32) []cluster.Response {
		resps := []cluster.Response{ans("a", .9, pair...), ans("b", .9, pair...), ans("c", .9, right...), ans("d", .9, right...)}
		for _, id := range []string{"x", "y", "z"} {
			resps = append(resps, ans(id, .9, lie...))
		}
		return resps
	}
	early, late := seq(500, 16), splice(right, 8, 500)
	cases := []struct {
		name  string
		lie   []int32
		resps []cluster.Response
	}{
		// a and b are right for 8 tokens, then err together: the majority is at the first token.
		{"shared prefix", early, against(splice(right, 8, 9000), early)},
		// Four nodes give the right answer outright.
		{"whole answer", early, against(right, early)},
		// Everyone agrees for 8 tokens, then x, y, z lie: the majority is at the ninth.
		{"late lie", late, against(right, late)},
	}

	a := agg(t, "adaptive").(tracked)
	twinsHistory(a)
	if got := fmt.Sprint(a.Tracker().Blocs()); got != fmt.Sprint([][]string{{"a", "b"}}) {
		t.Fatalf("setup: blocs %s, want the pair [a b]", got)
	}
	if a.Trust("a") >= a.Trust("c") {
		t.Fatalf("setup: trust a=%.2f c=%.2f", a.Trust("a"), a.Trust("c"))
	}
	for _, c := range cases {
		if d := a.Aggregate(cluster.Request{ID: c.name}, c.resps); !cluster.Equal(d.Tokens, right) {
			t.Errorf("%s: three nodes out-voted four, winners %v", c.name, d.Winners)
		}
	}

	// Without the rule, the same history gives both votes away.
	unbounded := agg(t, "adaptive-unbounded")
	twinsHistory(unbounded)
	for _, c := range cases {
		if d := unbounded.Aggregate(cluster.Request{ID: c.name}, c.resps); !cluster.Equal(d.Tokens, c.lie) {
			t.Errorf("%s: expected the unbounded variant to lose this vote, winners %v", c.name, d.Winners)
		}
	}
}

// A screened answer has been set aside: it is not a backer, and a majority
// made of screened answers is not a majority. Here two nodes return half
// the right answer (flagged truncated), two return all of it but report
// more confidence than the rest (flagged over-confident), and three share
// an early mistake. Whatever is returned must have been given by a node
// whose vote counted.
func TestAScreenedAnswerIsNotABacker(t *testing.T) {
	right := seq(1, 16)
	wrong := splice(right, 3, 500)
	resps := []cluster.Response{
		ans("t1", .8, right[:8]...), ans("t2", .8, right[:8]...),
		ans("r1", .9, right...), ans("r2", .9, right...),
		ans("w1", .8, wrong...), ans("w2", .8, wrong...), ans("w3", .8, wrong...),
	}
	d := agg(t, "adaptive").Aggregate(cluster.Request{ID: "r"}, resps)
	if fmt.Sprint(d.Suspects) != "[r1 r2 t1 t2]" {
		t.Fatalf("setup: expected t1, t2, r1, r2 to be screened, suspects %v", d.Suspects)
	}
	if fmt.Sprint(d.Winners) != "[w1 w2 w3]" || d.Share != 1 {
		t.Fatalf("winners %v with share %.3f: the answer returned was backed only by nodes that had been set aside", d.Winners, d.Share)
	}
}

// The price of making a majority final: it is taken at its word even when
// part of it is a bloc the strategy has already identified. x, y, z collude
// on every fourth request and are learned as a bloc; then honest d makes a
// mistake and the bloc gives the very same answer. Four of seven nodes now
// agree, and that answer is returned, although three nodes with clean
// records are right. The unbounded variant, which trusts what it has
// inferred, gets this one right — and is the one that can be captured.
func TestAMajorityIsTakenAtItsWord(t *testing.T) {
	right := seq(1, 16)
	wrong := splice(right, 8, 9000)
	history := func(a cluster.Aggregator) {
		for i := 0; i < 200; i++ {
			var resps []cluster.Response
			for _, id := range []string{"x", "y", "z"} {
				tokens := right
				if i%4 == 0 {
					tokens = splice(right, 8, 7000+i)
				}
				resps = append(resps, ans(id, .9, tokens...))
			}
			for _, id := range []string{"a", "b", "c", "d"} {
				resps = append(resps, ans(id, .9, right...))
			}
			a.Aggregate(cluster.Request{ID: fmt.Sprint(i)}, resps)
		}
	}
	final := []cluster.Response{
		ans("a", .9, right...), ans("b", .9, right...), ans("c", .9, right...), ans("d", .9, wrong...),
		ans("x", .9, wrong...), ans("y", .9, wrong...), ans("z", .9, wrong...),
	}

	a := agg(t, "adaptive").(tracked)
	history(a)
	if got := fmt.Sprint(a.Tracker().Blocs()); got != fmt.Sprint([][]string{{"x", "y", "z"}}) {
		t.Fatalf("setup: blocs %s, want [x y z]", got)
	}
	if d := a.Aggregate(cluster.Request{ID: "final"}, final); !cluster.Equal(d.Tokens, wrong) {
		t.Fatalf("expected the four-node majority to stand, winners %v", d.Winners)
	}

	unbounded := agg(t, "adaptive-unbounded")
	history(unbounded)
	if d := unbounded.Aggregate(cluster.Request{ID: "final"}, final); !cluster.Equal(d.Tokens, right) {
		t.Fatalf("expected the unbounded variant to side with the three clean records, winners %v", d.Winners)
	}
}

// The scenario that exposed it, end to end: two honest replicas with
// common-mode errors and three sleepers. Until a majority was made final,
// adaptive answered with the sleepers on nearly every request after they
// woke — and, since none of those decisions was decisive, never learned
// otherwise.
func TestCorrelatedReplicasDoNotHandSleepersTheVote(t *testing.T) {
	for _, c := range []struct {
		name       string
		good, corr int     // honest replicas that err on their own; honest replicas that share their mistakes
		floor      float64 // what adaptive must reach
	}{
		// Two of the four honest replicas share their mistakes. When they
		// err the other two still out-vote them token by token, so nearly
		// every request is recoverable.
		{"twins", 2, 2, 0.9},
		// Three of the four do. When they err, three honest nodes are wrong
		// together and no vote can tell: about a fifth of requests are lost
		// to every strategy. The claim is only that adaptive is not captured.
		{"triplets", 1, 3, 0.7},
	} {
		for seed := uint64(1); seed <= 3; seed++ {
			sc := dataset.Scenario{Name: c.name, Seed: seed, Requests: 1000, Groups: []dataset.Group{
				{Count: c.good, Noise: 0.05}, {Count: c.corr, Noise: 0.2, Corr: 1}, {Count: 3, Preset: "sleeper"},
			}}
			recs, _, err := dataset.Collect(context.Background(), sc)
			if err != nil {
				t.Fatal(err)
			}
			got := replay(t, agg(t, "adaptive"), recs, 0, len(recs))
			unbounded := replay(t, agg(t, "adaptive-unbounded"), recs, 0, len(recs))
			maj, tv := accuracy(t, recs, "majority"), accuracy(t, recs, "token-vote")
			t.Logf("%s seed %d: adaptive=%.3f adaptive-unbounded=%.3f token-vote=%.3f majority=%.3f", c.name, seed, got, unbounded, tv, maj)
			if got < c.floor || got < tv-0.01 {
				t.Errorf("%s seed %d: adaptive %.3f, want at least %.2f and no worse than token-vote's %.3f", c.name, seed, got, c.floor, tv)
			}
			if unbounded > 0.5 {
				t.Errorf("%s seed %d: expected the unbounded variant to be captured, got %.3f", c.name, seed, unbounded)
			}
		}
	}
}

// Nothing a strategy returns may depend on the order the responses are
// listed in, which is the order they happened to arrive in. Screening is
// where this is easy to get wrong: it compares each node with its peers
// against thresholds, and unless the arithmetic is done in a fixed order,
// rounding alone moves a borderline answer across one.
func TestNothingDependsOnArrivalOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cfg := detect.Default()
	for _, attack := range []string{"collude", "perturb"} {
		recs := load(t, attack, 7, 2, 1000)
		for _, name := range Names() {
			listed, shuffled := agg(t, name), agg(t, name)
			for i, rec := range recs {
				resps := cluster.Blind(rec.Responses)
				other := append([]cluster.Response(nil), resps...)
				rng.Shuffle(len(other), func(i, j int) { other[i], other[j] = other[j], other[i] })
				if a, b := fmt.Sprint(cfg.Inspect(resps)), fmt.Sprint(cfg.Inspect(other)); a != b {
					t.Fatalf("%s request %d: flags %s, but %s with the responses in another order", attack, i, a, b)
				}
				d, o := listed.Aggregate(rec.Request, resps), shuffled.Aggregate(rec.Request, other)
				same := cluster.Equal(d.Tokens, o.Tokens) && d.Abstained == o.Abstained &&
					fmt.Sprint(d.Winners) == fmt.Sprint(o.Winners) && fmt.Sprint(d.Suspects) == fmt.Sprint(o.Suspects) &&
					math.Abs(d.Share-o.Share) < 1e-9 && math.Abs(d.Margin-o.Margin) < 1e-9
				if !same {
					t.Fatalf("%s, %s, request %d: decision %+v, but %+v with the responses in another order", attack, name, i, d, o)
				}
			}
		}
	}
}

// Two nodes that give the same answer get the same verdict on it. The
// flags that depend only on the answer — outlier, minority, truncated —
// may not single out one of two identical answers.
func TestIdenticalAnswersGetTheSameVerdict(t *testing.T) {
	cfg := detect.Default()
	for seed := uint64(1); seed <= 5; seed++ {
		sc, err := dataset.Standard("truncate", 7, 2, 1000, seed)
		if err != nil {
			t.Fatal(err)
		}
		recs, _, err := dataset.Collect(context.Background(), sc)
		if err != nil {
			t.Fatal(err)
		}
		c := struct {
			attack string
			f      int
		}{sc.Name, 2}
		for i, rec := range recs {
			verdict := map[string]string{} // node -> its answer-only flags
			for _, f := range cfg.Inspect(cluster.Blind(rec.Responses)) {
				if f.Reason == detect.Outlier || f.Reason == detect.Minority || f.Reason == detect.Truncated {
					verdict[f.Node] += f.Reason + " "
				}
			}
			byAnswer := map[string]string{} // answer -> first node seen giving it
			for _, r := range rec.Responses {
				if !r.OK() {
					continue
				}
				k := cluster.Key(r.Tokens)
				if first, ok := byAnswer[k]; !ok {
					byAnswer[k] = r.Node
				} else if verdict[first] != verdict[r.Node] {
					t.Fatalf("%s f=%d request %d: %s and %s gave the same answer but were flagged [%s] and [%s]",
						c.attack, c.f, i, first, r.Node, verdict[first], verdict[r.Node])
				}
			}
		}
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
