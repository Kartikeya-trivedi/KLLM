package detect

import (
	"fmt"
	"testing"

	"kllm/cluster"
)

func seq(from, n int) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = int32(from + i)
	}
	return s
}

// ans is a 16-token answer starting at `from`, with flat confidence and a latency.
func ans(node string, from int, conf float32, ms float64) cluster.Response {
	c := make([]float32, 16)
	for i := range c {
		c[i] = conf
	}
	return cluster.Response{Node: node, Tokens: seq(from, 16), Conf: c, LatencyMs: ms}
}

func reasons(flags []Flag) string {
	s := ""
	for _, f := range flags {
		s += f.Node + ":" + f.Reason + " "
	}
	return s
}

func TestInspectCleanRequestRaisesNothing(t *testing.T) {
	resps := []cluster.Response{
		ans("a", 1, .85, 100), ans("b", 1, .84, 110), ans("c", 1, .86, 95), ans("d", 1, .85, 105), ans("e", 1, .83, 120),
	}
	if flags := Default().Inspect(resps); len(flags) != 0 {
		t.Fatalf("clean request flagged: %s", reasons(flags))
	}
}

func TestInspectEachSignal(t *testing.T) {
	base := func() []cluster.Response {
		return []cluster.Response{
			ans("a", 1, .85, 100), ans("b", 1, .84, 110), ans("c", 1, .86, 95), ans("d", 1, .85, 105),
		}
	}
	oneOff := ans("x", 1, .85, 100)
	oneOff.Tokens = append(seq(1, 15), 999) // a single wrong token at the end
	short := ans("x", 1, .85, 100)
	short.Tokens, short.Conf = short.Tokens[:8], short.Conf[:8]

	cases := []struct {
		name string
		x    cluster.Response
		want string
	}{
		{"fabricated answer", ans("x", 500, .85, 100), "x:minority x:outlier "},
		{"one wrong token is a dissent, not an outlier", oneOff, "x:minority "},
		{"claimed certainty", ans("x", 1, .999, 100), "x:overconfident "},
		{"answered before the model could have", ans("x", 1, .85, 2), "x:fast "},
		{"stalled", ans("x", 1, .85, 450), "x:slow "},
		{"half an answer", short, "x:minority x:outlier x:truncated "},
		{"crashed", cluster.Response{Node: "x", Err: "connection refused"}, "x:failed "},
		{"cancelled by the orchestrator", cluster.Response{Node: "x", Err: cluster.ErrCancelled}, ""},
	}
	for _, c := range cases {
		if got := reasons(Default().Inspect(append(base(), c.x))); got != c.want {
			t.Errorf("%s: flagged %q, want %q", c.name, got, c.want)
		}
	}
}

func TestInspectNeedsPeers(t *testing.T) {
	// Two answers that disagree: there is no "rest" to be unlike.
	resps := []cluster.Response{ans("a", 1, .85, 100), ans("b", 500, .999, 2)}
	if flags := Default().Inspect(resps); len(flags) != 0 {
		t.Fatalf("flagged with too few peers: %s", reasons(flags))
	}
}

func TestAdversarialSeparatesMisbehaviourFromBadLuck(t *testing.T) {
	for reason, want := range map[string]bool{
		Outlier: true, Overconfident: true, Fast: true, Truncated: true, Failed: true,
		Minority: false, Slow: false,
	} {
		if Adversarial(reason) != want {
			t.Errorf("Adversarial(%s) = %v", reason, !want)
		}
	}
}

// round feeds the tracker one request in which each listed node answers
// with the given "version" of the answer; version 0 is the consensus.
func round(tr *Tracker, versions map[string]int) {
	var resps []cluster.Response
	for id, v := range versions {
		resps = append(resps, ans(id, 1+1000*v, .85, 100))
	}
	tr.Observe(resps, seq(1, 16))
}

func TestTrustFollowsAgreement(t *testing.T) {
	tr := NewTracker(Default())
	if tr.Trust("stranger") != 0.5 {
		t.Fatal("an unknown node should start at 0.5")
	}
	for i := 0; i < 200; i++ {
		lossy := 0
		if i%5 == 0 { // wrong one time in five
			lossy = 1 + i // ...in its own way each time
		}
		round(tr, map[string]int{"good": 0, "lossy": lossy, "bad": 7, "also-good": 0})
	}
	good, lossy, bad := tr.Trust("good"), tr.Trust("lossy"), tr.Trust("bad")
	if good < 0.95 || bad > 0.05 || lossy < 0.7 || lossy > 0.9 {
		t.Fatalf("trust good=%.2f lossy=%.2f bad=%.2f", good, lossy, bad)
	}
	sus := tr.Suspects()
	if len(sus) != 1 || sus[0].Node != "bad" {
		t.Fatalf("suspects: %+v", sus)
	}
	// A node that is wrong from the start trips the drift alarm within a
	// dozen requests — before it has been seen often enough for low trust
	// to count — and is untrusted from then on.
	why := fmt.Sprint(sus[0].Reasons)
	if (why != "[drift untrusted]" && why != "[untrusted]") || sus[0].DetectedAt > 12 || sus[0].DetectedAt >= Default().MinSeen {
		t.Fatalf("bad node: %+v", sus[0])
	}
}

func TestDriftCatchesASleeperLongBeforeItsTrustDecays(t *testing.T) {
	tr := NewTracker(Default())
	for i := 0; i < 300; i++ {
		round(tr, map[string]int{"a": 0, "b": 0, "c": 0, "sleeper": 0})
	}
	if len(tr.Suspects()) != 0 {
		t.Fatalf("nobody has misbehaved yet: %+v", tr.Suspects())
	}
	for i := 0; i < 12; i++ {
		round(tr, map[string]int{"a": 0, "b": 0, "c": 0, "sleeper": 3})
	}
	sus := tr.Suspects()
	if len(sus) != 1 || sus[0].Node != "sleeper" || fmt.Sprint(sus[0].Reasons) != "[drift]" {
		t.Fatalf("suspects after 12 bad answers: %+v", sus)
	}
	if woke := sus[0].DetectedAt - 300; woke < 1 || woke > 10 {
		t.Fatalf("detected %d requests after waking", woke)
	}
	if got := tr.Trust("sleeper"); got < Default().TrustFloor+0.2 {
		t.Fatalf("trust alone should still be well above the floor, but reads %.2f", got)
	}
}

func TestHonestNoiseDoesNotTripTheDriftAlarm(t *testing.T) {
	tr := NewTracker(Default())
	state := uint64(12345)
	next := func() float64 { // xorshift: a fixed pseudo-random stream
		state ^= state << 13
		state ^= state >> 7
		state ^= state << 17
		return float64(state>>11) / (1 << 53)
	}
	for i := 0; i < 5000; i++ {
		v := map[string]int{"a": 0, "b": 0, "c": 0, "lossy": 0, "lossier": 0}
		if next() < 0.15 {
			v["lossy"] = 1 + i
		}
		if next() < 0.28 {
			v["lossier"] = 100000 + i
		}
		round(tr, v)
	}
	if sus := tr.Suspects(); len(sus) != 0 {
		t.Fatalf("honest-but-lossy nodes flagged: %+v", sus)
	}
}

func TestBlocsAreNodesThatDeviateTogether(t *testing.T) {
	tr := NewTracker(Default())
	for i := 0; i < 200; i++ {
		v := map[string]int{"a": 0, "b": 0, "c": 0, "d": 0, "x": 0, "y": 0, "z": 0}
		if i%4 == 0 { // the bloc strikes on a quarter of requests, in unison
			v["x"], v["y"], v["z"] = 9, 9, 9
		}
		if i%50 == 1 { // two honest nodes take the same wrong turn now and then
			v["a"], v["b"] = 5, 5
		}
		if i%7 == 0 { // and one errs alone
			v["c"] = 1 + i
		}
		round(tr, v)
	}
	if got := fmt.Sprint(tr.Blocs()); got != "[[x y z]]" {
		t.Fatalf("blocs = %s", got)
	}
	// Each member is right three times in four — individually it looks
	// like a mediocre honest node. Only the lockstep gives it away.
	for _, s := range tr.Suspects() {
		if s.Node != "x" && s.Node != "y" && s.Node != "z" {
			t.Fatalf("honest node %s flagged: %+v", s.Node, s)
		}
		if fmt.Sprint(s.Reasons) != "[collusion]" || s.Trust < 0.6 {
			t.Fatalf("bloc member: %+v", s)
		}
	}
	if len(tr.Suspects()) != 3 || tr.Observed() != 200 {
		t.Fatalf("suspects: %+v", tr.Suspects())
	}
}
