package node

import (
	"context"
	"fmt"
	"testing"
	"time"

	"kllm/cluster"
)

func prompts(n int) []cluster.Request {
	reqs := make([]cluster.Request, n)
	for i := range reqs {
		p := make([]int32, 4+i%9)
		for j := range p {
			p[j] = int32(mix(uint64(i*131+j)) % 512)
		}
		reqs[i] = cluster.Request{ID: fmt.Sprintf("r%d", i), Prompt: p, MaxNewTokens: 16}
	}
	return reqs
}

func gen(t *testing.T, b Backend, r cluster.Request) Output {
	t.Helper()
	out, err := b.Generate(context.Background(), r)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return out
}

func TestNoiselessReplicaReproducesReference(t *testing.T) {
	m := DefaultSimModel(7)
	s := NewSim(m, 99, 0)
	for _, r := range prompts(50) {
		want, wantConf := m.Reference(r.Prompt, r.MaxNewTokens)
		got := gen(t, s, r)
		if !cluster.Equal(got.Tokens, want) {
			t.Fatalf("noise=0 replica diverged from the reference on %s", r.ID)
		}
		for i := range wantConf {
			if got.Conf[i] != wantConf[i] {
				t.Fatalf("confidence mismatch at step %d", i)
			}
		}
		for _, tok := range got.Tokens {
			if tok < 0 || int(tok) >= m.Vocab {
				t.Fatalf("token %d outside vocab", tok)
			}
		}
	}
}

func TestSimIsDeterministicAndSeedSensitive(t *testing.T) {
	m := DefaultSimModel(7)
	r := prompts(1)[0]
	a := gen(t, NewSim(m, 1, 0.3), r)
	b := gen(t, NewSim(m, 1, 0.3), r)
	if !cluster.Equal(a.Tokens, b.Tokens) || a.Latency != b.Latency {
		t.Fatal("same seeds, same request, different output")
	}
	other, _ := DefaultSimModel(8).Reference(r.Prompt, r.MaxNewTokens)
	ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
	if cluster.Equal(other, ref) {
		t.Fatal("a different model seed should give a different reference")
	}
}

// The property the whole lab leans on: a noisy replica is right most of the
// time, wrong some of the time, and reports low confidence where it erred.
func TestNoisyReplicaErrsAtLowConfidence(t *testing.T) {
	m := DefaultSimModel(3)
	s := NewSim(m, 5, 0.1)
	reqs := prompts(2000)
	exact := 0
	var flipConf, keepConf float64
	var flips, keeps int
	for _, r := range reqs {
		ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
		got := gen(t, s, r)
		if cluster.Equal(got.Tokens, ref) {
			exact++
		}
		k := cluster.LCP(got.Tokens, ref)
		for i := 0; i < k; i++ {
			keepConf += float64(got.Conf[i])
			keeps++
		}
		if k < len(ref) { // first divergent step is the flip itself
			flipConf += float64(got.Conf[k])
			flips++
		}
	}
	acc := float64(exact) / float64(len(reqs))
	if acc < 0.75 || acc > 0.93 {
		t.Fatalf("noise=0.1 exact-match = %.3f, want roughly 0.85", acc)
	}
	if flips == 0 {
		t.Fatal("no divergences observed")
	}
	if fc, kc := flipConf/float64(flips), keepConf/float64(keeps); fc >= kc-0.2 {
		t.Fatalf("flipped tokens should be much less confident: flip=%.3f keep=%.3f", fc, kc)
	}
}

// Independent replicas disagree with each other when they err; fully
// correlated ones err identically — the case voting cannot fix.
func TestCorrelationControlsSharedErrors(t *testing.T) {
	m := DefaultSimModel(11)
	reqs := prompts(1500)
	agreeWhenWrong := func(corr float64) float64 {
		a, b := NewSim(m, 1, 0.2), NewSim(m, 2, 0.2)
		a.Corr, b.Corr = corr, corr
		var bothWrong, same int
		for _, r := range reqs {
			ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
			x, y := gen(t, a, r), gen(t, b, r)
			if !cluster.Equal(x.Tokens, ref) && !cluster.Equal(y.Tokens, ref) {
				bothWrong++
				if cluster.Equal(x.Tokens, y.Tokens) {
					same++
				}
			}
		}
		if bothWrong == 0 {
			t.Fatalf("corr=%g: no joint errors to measure", corr)
		}
		return float64(same) / float64(bothWrong)
	}
	if got := agreeWhenWrong(1); got != 1 {
		t.Fatalf("corr=1 replicas must err identically, agreed on %.3f of joint errors", got)
	}
	if got := agreeWhenWrong(0); got > 0.9 {
		t.Fatalf("corr=0 replicas should not err identically every time (%.3f)", got)
	}
}

func TestLatencyModel(t *testing.T) {
	s := NewSim(DefaultSimModel(1), 4, 0.1)
	var sum time.Duration
	var stalls int
	reqs := prompts(3000)
	for _, r := range reqs {
		d := gen(t, s, r).Latency
		if d <= 0 {
			t.Fatal("non-positive latency")
		}
		if d > 250*time.Millisecond {
			stalls++
		}
		sum += d
	}
	mean := sum / time.Duration(len(reqs))
	if mean < 80*time.Millisecond || mean > 140*time.Millisecond {
		t.Fatalf("mean modelled latency %v, want ~100ms", mean)
	}
	if frac := float64(stalls) / float64(len(reqs)); frac < 0.005 || frac > 0.06 {
		t.Fatalf("heavy-tail fraction %.3f, want ~0.02", frac)
	}
}

func TestSimRejectsBadRequests(t *testing.T) {
	s := NewSim(DefaultSimModel(1), 4, 0)
	if _, err := s.Generate(context.Background(), cluster.Request{MaxNewTokens: 4}); err == nil {
		t.Fatal("empty prompt accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.Generate(ctx, prompts(1)[0]); err == nil {
		t.Fatal("cancelled context ignored")
	}
}
