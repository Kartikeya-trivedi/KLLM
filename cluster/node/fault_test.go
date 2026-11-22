package node

import (
	"context"
	"errors"
	"testing"
	"time"

	"kllm/cluster"
)

func preset(t *testing.T, name string) Fault {
	t.Helper()
	f, ok := Preset(name)
	if !ok {
		t.Fatalf("no preset %q", name)
	}
	return f
}

// omniscient adversaries run a noiseless replica: they know the right
// answer and deviate on purpose.
func adversary(m SimModel, nodeSeed uint64, f Fault) Backend {
	return Inject(NewSim(m, nodeSeed, 0), f, nodeSeed, m.Vocab)
}

// counting records how often the model behind a fault was actually run,
// and says whether running it is free.
type counting struct {
	Backend
	instant bool
	calls   int
}

func (c *counting) Instant() bool { return c.instant }

func (c *counting) Generate(ctx context.Context, req cluster.Request) (Output, error) {
	c.calls++
	return c.Backend.Generate(ctx, req)
}

func TestHonestProfileIsPassthrough(t *testing.T) {
	m := DefaultSimModel(1)
	inner := NewSim(m, 3, 0.1)
	if Inject(inner, preset(t, "none"), 3, m.Vocab) != Backend(inner) {
		t.Fatal("an honest profile should not wrap the backend")
	}
}

func TestCrashAndFlaky(t *testing.T) {
	m := DefaultSimModel(1)
	reqs := prompts(1000)

	b := adversary(m, 1, preset(t, "crash"))
	out, err := b.Generate(context.Background(), reqs[0])
	if !errors.Is(err, ErrCrashed) || !out.Label.Attacked || out.Label.Fault != "crash" {
		t.Fatalf("crash: err=%v label=%+v", err, out.Label)
	}

	b = adversary(m, 1, preset(t, "flaky"))
	failed := 0
	for _, r := range reqs {
		out, err := b.Generate(context.Background(), r)
		if (err != nil) != out.Label.Attacked {
			t.Fatal("label disagrees with outcome")
		}
		if err != nil {
			failed++
		}
	}
	if frac := float64(failed) / float64(len(reqs)); frac < 0.24 || frac > 0.36 {
		t.Fatalf("flaky failed %.3f of requests, want ~0.3", frac)
	}
}

func TestStragglerIsHonestButLate(t *testing.T) {
	m := DefaultSimModel(1)
	r := prompts(1)[0]
	honest := gen(t, NewSim(m, 1, 0), r)
	late := gen(t, adversary(m, 1, preset(t, "straggler")), r)
	if !cluster.Equal(honest.Tokens, late.Tokens) || late.Label.Attacked {
		t.Fatal("a straggler must not change the answer")
	}
	if got := late.Latency - honest.Latency; got != 400*time.Millisecond {
		t.Fatalf("straggler delay = %v", got)
	}
}

func TestRandomIsUncoordinated(t *testing.T) {
	m := DefaultSimModel(1)
	r := prompts(1)[0]
	ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
	a := gen(t, adversary(m, 1, preset(t, "random")), r)
	b := gen(t, adversary(m, 2, preset(t, "random")), r)
	if cluster.Equal(a.Tokens, ref) || cluster.Equal(a.Tokens, b.Tokens) {
		t.Fatal("random outputs should match neither the truth nor each other")
	}
	if len(a.Tokens) != len(ref) || !a.Label.Attacked {
		t.Fatalf("len=%d label=%+v", len(a.Tokens), a.Label)
	}
	for _, tok := range a.Tokens {
		if tok < 0 || int(tok) >= m.Vocab {
			t.Fatalf("token %d outside vocab", tok)
		}
	}
}

func TestPerturbIsSubtle(t *testing.T) {
	m := DefaultSimModel(1)
	for _, r := range prompts(200) {
		ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
		got := gen(t, adversary(m, 1, preset(t, "perturb")), r)
		if len(got.Tokens) != len(ref) || cluster.Equal(got.Tokens, ref) {
			t.Fatal("perturb must change at least one token and keep the length")
		}
		if cluster.Similarity(got.Tokens, ref) < 0.5 {
			t.Fatal("perturb changed too much to be subtle")
		}
	}
}

func TestColludersAgreeOnAWrongAnswer(t *testing.T) {
	m := DefaultSimModel(1)
	for _, r := range prompts(200) {
		ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
		a := gen(t, adversary(m, 1, preset(t, "collude")), r)
		b := gen(t, adversary(m, 2, preset(t, "collude")), r)
		if !cluster.Equal(a.Tokens, b.Tokens) {
			t.Fatal("members of one group must emit the same answer")
		}
		if cluster.Equal(a.Tokens, ref) {
			t.Fatal("the colluded answer must be wrong")
		}
		other := preset(t, "collude")
		other.Secret = 2
		if c := gen(t, adversary(m, 3, other), r); cluster.Equal(a.Tokens, c.Tokens) {
			t.Fatal("different groups should not agree")
		}
	}
}

func TestStealthKeepsAnHonestPrefix(t *testing.T) {
	m := DefaultSimModel(1)
	for _, r := range prompts(200) {
		ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
		a := gen(t, adversary(m, 1, preset(t, "stealth")), r)
		b := gen(t, adversary(m, 2, preset(t, "stealth")), r)
		if !cluster.Equal(a.Tokens, b.Tokens) {
			t.Fatal("stealth colluders must still agree")
		}
		if got := cluster.LCP(a.Tokens, ref); got != 12 {
			t.Fatalf("keep=0.75 of 16 tokens should diverge exactly at 12, got %d", got)
		}
	}
}

func TestIntermittentGroupStrikesTogether(t *testing.T) {
	m := DefaultSimModel(1)
	a := adversary(m, 1, preset(t, "intermittent"))
	b := adversary(m, 2, preset(t, "intermittent"))
	reqs := prompts(1000)
	struck := 0
	for _, r := range reqs {
		x, y := gen(t, a, r), gen(t, b, r)
		if x.Label.Attacked != y.Label.Attacked {
			t.Fatal("colluders chose different targets")
		}
		ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
		if x.Label.Attacked == cluster.Equal(x.Tokens, ref) {
			t.Fatal("label disagrees with whether the answer was tampered")
		}
		if x.Label.Attacked {
			struck++
		}
	}
	if frac := float64(struck) / float64(len(reqs)); frac < 0.24 || frac > 0.36 {
		t.Fatalf("struck %.3f of requests, want ~0.3", frac)
	}
}

func TestSleeperWaits(t *testing.T) {
	m := DefaultSimModel(1)
	f := preset(t, "sleeper")
	f.After = 10
	b := adversary(m, 1, f)
	for i, r := range prompts(30) {
		ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
		got := gen(t, b, r)
		wantAttack := i >= 10
		if got.Label.Attacked != wantAttack || cluster.Equal(got.Tokens, ref) == wantAttack {
			t.Fatalf("request %d: attacked=%v", i, got.Label.Attacked)
		}
		if got.Label.Fault != "sleeper" {
			t.Fatalf("label %q", got.Label.Fault)
		}
	}
}

func TestSpoofFastAndTruncate(t *testing.T) {
	m := DefaultSimModel(1)
	r := prompts(1)[0]

	spoof := gen(t, adversary(m, 1, preset(t, "overconfident")), r)
	for _, c := range spoof.Conf {
		if c != 0.999 {
			t.Fatalf("spoofed confidence = %g", c)
		}
	}
	if len(spoof.Conf) != len(spoof.Tokens) {
		t.Fatal("confidence length mismatch")
	}

	// A fast colluder differs from an ordinary one in arrival time and in
	// nothing else — in particular not in the confidence it reports, or the
	// `fast` condition would measure a confidence attack as well.
	fast := gen(t, adversary(m, 1, preset(t, "fast")), r)
	if fast.Latency != fastLatency || len(fast.Tokens) != r.MaxNewTokens {
		t.Fatalf("fast: latency=%v len=%d", fast.Latency, len(fast.Tokens))
	}
	slow := gen(t, adversary(m, 1, preset(t, "collude")), r)
	if !cluster.Equal(fast.Tokens, slow.Tokens) {
		t.Fatal("fast and ordinary colluders of one group must agree")
	}
	if slow.Latency <= fastLatency {
		t.Fatalf("setup: the ordinary colluder answered in %v", slow.Latency)
	}
	if len(fast.Conf) != len(slow.Conf) {
		t.Fatalf("fast reports %d confidences, the ordinary colluder %d", len(fast.Conf), len(slow.Conf))
	}
	for i := range fast.Conf {
		if fast.Conf[i] != slow.Conf[i] {
			t.Fatalf("token %d: fast reports confidence %g, the ordinary colluder %g", i, fast.Conf[i], slow.Conf[i])
		}
	}

	// A fast adversary skips the model only when waiting for it would cost
	// real time and its answer does not need it.
	for _, c := range []struct {
		name    string
		fault   Fault
		instant bool // is the model behind it free to consult?
		calls   int
	}{
		{"synthetic model", preset(t, "fast"), true, 1},
		{"live engine", preset(t, "fast"), false, 0},
		{"live engine, declared confidence", Fault{Kind: FaultCollude, Secret: 1, Fast: true, SpoofConf: 0.95}, false, 0},
		{"live engine, honest prefix", Fault{Kind: FaultCollude, Secret: 1, Fast: true, Keep: 0.5}, false, 1},
	} {
		inner := &counting{Backend: NewSim(m, 1, 0), instant: c.instant}
		out := gen(t, Inject(inner, c.fault, 1, m.Vocab), r)
		if inner.calls != c.calls {
			t.Errorf("%s: the model ran %d times, want %d", c.name, inner.calls, c.calls)
		}
		if out.Latency != fastLatency || len(out.Tokens) != r.MaxNewTokens || len(out.Conf) != len(out.Tokens) {
			t.Errorf("%s: latency=%v tokens=%d conf=%d", c.name, out.Latency, len(out.Tokens), len(out.Conf))
		}
	}

	// With or without the model, a group's fabricated lie is the same lie:
	// a member that answered blind must not split the group's vote. A small
	// vocabulary makes the case that used to differ (the lie's first token
	// happening to be the honest one) common enough to meet.
	small := SimModel{Seed: 1, Vocab: 8, HardFrac: 0.1}
	for _, p := range prompts(200) {
		blind := gen(t, Inject(&counting{Backend: NewSim(small, 1, 0)}, preset(t, "fast"), 1, small.Vocab), p)
		sighted := gen(t, Inject(NewSim(small, 2, 0), preset(t, "collude"), 2, small.Vocab), p)
		if !cluster.Equal(blind.Tokens, sighted.Tokens) {
			t.Fatalf("%s: a member that skipped the model lied %v, one that ran it %v", p.ID, blind.Tokens, sighted.Tokens)
		}
		for _, c := range blind.Conf {
			if c != 0.9 {
				t.Fatalf("%s: with no model to copy and no declared confidence, reported %g", p.ID, c)
			}
		}
	}

	ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
	short := gen(t, adversary(m, 1, preset(t, "truncate")), r)
	if len(short.Tokens) != len(ref)/2 || !cluster.Equal(short.Tokens, ref[:len(ref)/2]) {
		t.Fatalf("truncate returned %d tokens", len(short.Tokens))
	}
}

func TestParseFault(t *testing.T) {
	if f, err := ParseFault(""); err != nil || !f.Honest() {
		t.Fatalf("empty: %+v %v", f, err)
	}
	if f, err := ParseFault("stealth"); err != nil || f.Keep != 0.75 || f.Name != "stealth" {
		t.Fatalf("preset: %+v %v", f, err)
	}
	f, err := ParseFault(`{"kind":"collude","secret":9,"rate":0.5}`)
	if err != nil || f.Secret != 9 || f.Rate != 0.5 {
		t.Fatalf("json: %+v %v", f, err)
	}
	for _, bad := range []string{"nope", `{"kind":"nope"}`, `{"kind":"collude","rate":2}`, `{`} {
		if _, err := ParseFault(bad); err == nil {
			t.Errorf("ParseFault(%q) accepted", bad)
		}
	}
	for _, name := range PresetNames() {
		if f, _ := Preset(name); f.validate() != nil {
			t.Errorf("preset %q is invalid", name)
		}
	}
}
