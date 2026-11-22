package server

import (
	"context"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"kllm/cluster"
	"kllm/cluster/node"
	"kllm/engine"
	"kllm/engine/oracle"
)

// The real engine as a cluster node: /v1/infer must return exactly what a
// direct greedy decode returns, plus a well-formed confidence per token,
// including when several requests share GPU batches. Skips without the
// built backend and the tiny test model (see docs/SETUP.md).
func TestInferServesTheEngineAsAClusterNode(t *testing.T) {
	// oracle.BackendLib is relative to engine/<pkg>; this package is one level up.
	dll := filepath.Join("..", "build", filepath.Base(oracle.BackendLib()))
	model := filepath.Join("..", "testmodels", "tiny-llama")
	for _, p := range []string{dll, model} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s — build the backend and run make_test_model.py first", p)
		}
	}
	e, err := engine.New(dll, model, engine.Options{MaxSeq: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()

	prompts := [][]int32{{1, 17, 42, 100, 7}, {3, 5, 8, 13, 21, 34, 55}, {9, 2}}
	const steps = 12
	want := make([][]int32, len(prompts))
	for i, p := range prompts {
		if want[i], err = e.Generate(p, steps); err != nil {
			t.Fatal(err)
		}
	}

	sched := engine.NewScheduler(e, 8)
	sched.Start()
	defer sched.Stop()
	srv := httptest.NewServer(HandlerAs(sched, Identity{Node: "gpu0", Model: "tiny-llama"}))
	defer srv.Close()
	c := node.NewHTTPClient("gpu0", srv.URL)

	var wg sync.WaitGroup
	got := make([]cluster.Response, 4*len(prompts))
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got[i] = c.Infer(context.Background(), cluster.Request{
				ID: fmt.Sprintf("r%d", i), Prompt: prompts[i%len(prompts)], MaxNewTokens: steps,
			})
		}()
	}
	wg.Wait()

	for i, r := range got {
		if !r.OK() {
			t.Fatalf("request %d: %s", i, r.Err)
		}
		if !cluster.Equal(r.Tokens, want[i%len(prompts)]) {
			t.Fatalf("request %d: /v1/infer returned %v, direct decode %v", i, r.Tokens, want[i%len(prompts)])
		}
		if len(r.Conf) != len(r.Tokens) {
			t.Fatalf("request %d: %d confidences for %d tokens", i, len(r.Conf), len(r.Tokens))
		}
		for _, p := range r.Conf {
			if !(p > 0 && p <= 1) {
				t.Fatalf("request %d: confidence %g is not a probability", i, p)
			}
		}
		if r.Model != "tiny-llama" || r.LatencyMs <= 0 {
			t.Fatalf("request %d: model=%q latency=%g", i, r.Model, r.LatencyMs)
		}
	}

	if r := c.Infer(context.Background(), cluster.Request{ID: "bad", MaxNewTokens: 4}); r.OK() {
		t.Fatal("an empty prompt must be rejected")
	}

	// The engine's /stats.json must be readable by the cluster's sampler.
	stats, err := c.Stats(context.Background())
	if err != nil || stats.Node != "gpu0" || stats.Model != "tiny-llama" {
		t.Fatalf("stats: %+v err=%v", stats, err)
	}
	if stats.Requests != uint64(len(got)) || stats.Tokens != uint64(len(got)*steps) || stats.InFlight != 0 {
		t.Fatalf("engine counters not visible to the sampler: %+v", stats)
	}
}

func TestArgmaxProbIsTheSoftmaxPeak(t *testing.T) {
	tok, p := engine.ArgmaxProb([]float32{0, 0, 0, 0})
	if tok != 0 || p < 0.2499 || p > 0.2501 {
		t.Fatalf("uniform logits: tok=%d p=%g, want 0 and 0.25", tok, p)
	}
	tok, p = engine.ArgmaxProb([]float32{-3, 9, 1})
	if tok != 1 || p < 0.999 {
		t.Fatalf("peaked logits: tok=%d p=%g", tok, p)
	}
	// Large logits must not overflow the exponentials.
	if _, p = engine.ArgmaxProb([]float32{1e4, 1e4 - 1}); p < 0.73 || p > 0.74 {
		t.Fatalf("large logits: p=%g, want sigmoid(1)=0.731", p)
	}
}
