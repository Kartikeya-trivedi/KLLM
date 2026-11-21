package node

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"kllm/cluster"
)

// quickSim is a replica whose modelled latency is small and exact, so
// timing assertions are tight without slow tests.
func quickSim(m SimModel, seed uint64, perToken time.Duration) *Sim {
	s := NewSim(m, seed, 0)
	s.Overhead, s.TokenTime, s.Jitter, s.TailProb = 0, perToken, 0, 0
	return s
}

func serve(t *testing.T, n *Node) *HTTPClient {
	t.Helper()
	srv := httptest.NewServer(n.Handler())
	t.Cleanup(srv.Close)
	return NewHTTPClient("as-registered", srv.URL)
}

func TestHTTPRoundTrip(t *testing.T) {
	m := DefaultSimModel(1)
	r := prompts(1)[0]
	want, _ := m.Reference(r.Prompt, r.MaxNewTokens)

	lab := serve(t, New(Config{ID: "n1", Labels: true}, NewSim(m, 1, 0)))
	got := lab.Infer(context.Background(), r)
	if !got.OK() || !cluster.Equal(got.Tokens, want) || len(got.Conf) != len(want) {
		t.Fatalf("response: %+v", got)
	}
	if got.Node != "as-registered" || got.RequestID != r.ID {
		t.Fatalf("identity not stamped by the client: node=%q id=%q", got.Node, got.RequestID)
	}
	if got.Label == nil || got.Label.Fault != FaultNone || got.Label.Attacked {
		t.Fatalf("lab label: %+v", got.Label)
	}

	prod := serve(t, New(Config{ID: "n2"}, NewSim(m, 1, 0)))
	if got := prod.Infer(context.Background(), r); got.Label != nil {
		t.Fatal("labels leaked from a node with Labels off")
	}
}

func TestVirtualTimeReportsWithoutWaiting(t *testing.T) {
	m := DefaultSimModel(1)
	r := prompts(1)[0] // 16 tokens
	n := New(Config{ID: "n"}, quickSim(m, 1, 50*time.Millisecond))
	start := time.Now()
	got := n.Infer(context.Background(), r)
	if wall := time.Since(start); wall > 300*time.Millisecond {
		t.Fatalf("virtual-time node actually waited %v", wall)
	}
	if got.LatencyMs < 800 || got.LatencyMs > 900 {
		t.Fatalf("reported latency %.1f ms, want ~800", got.LatencyMs)
	}
}

func TestRealtimeWaitsAndHonoursDeadline(t *testing.T) {
	m := DefaultSimModel(1)
	r := prompts(1)[0]
	c := serve(t, New(Config{ID: "n", Realtime: true}, quickSim(m, 1, 5*time.Millisecond))) // 80 ms

	start := time.Now()
	got := c.Infer(context.Background(), r)
	if wall := time.Since(start); !got.OK() || wall < 80*time.Millisecond {
		t.Fatalf("realtime node answered in %v (%+v)", wall, got.Err)
	}
	if got.LatencyMs < 80 {
		t.Fatalf("latency %.1f ms below the modelled 80", got.LatencyMs)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start = time.Now()
	got = c.Infer(ctx, r)
	if got.OK() || len(got.Tokens) != 0 {
		t.Fatal("a request past its deadline must fail")
	}
	if wall := time.Since(start); wall > 70*time.Millisecond {
		t.Fatalf("deadline not honoured: returned after %v", wall)
	}
}

func TestCrashSurfacesAsErrorWithLabel(t *testing.T) {
	m := DefaultSimModel(1)
	f, _ := Preset("crash")
	c := serve(t, New(Config{ID: "n", Labels: true}, Inject(NewSim(m, 1, 0), f, 1, m.Vocab)))
	got := c.Infer(context.Background(), prompts(1)[0])
	if got.OK() || !strings.Contains(got.Err, "crashed") {
		t.Fatalf("err = %q", got.Err)
	}
	if got.Label == nil || !got.Label.Attacked {
		t.Fatalf("crash label lost: %+v", got.Label)
	}
}

func TestLoadShedding(t *testing.T) {
	m := DefaultSimModel(1)
	n := New(Config{ID: "n", Realtime: true, MaxConcurrent: 1}, quickSim(m, 1, 5*time.Millisecond))
	c := serve(t, n)
	r := prompts(1)[0]

	var wg sync.WaitGroup
	results := make([]cluster.Response, 4)
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i] = c.Infer(context.Background(), r)
		}()
	}
	wg.Wait()
	ok, shed := 0, 0
	for _, res := range results {
		switch {
		case res.OK():
			ok++
		case res.Err == ErrOverloaded.Error():
			shed++
		}
	}
	if ok < 1 || shed < 1 || ok+shed != len(results) {
		t.Fatalf("ok=%d shed=%d of %d", ok, shed, len(results))
	}
	s, _ := n.Stats(context.Background())
	if s.Rejected != uint64(shed) || s.Requests != uint64(ok) || s.InFlight != 0 {
		t.Fatalf("stats: %+v", s)
	}
}

func TestStatsAndMetricsEndpoints(t *testing.T) {
	m := DefaultSimModel(1)
	n := New(Config{ID: "n7"}, NewSim(m, 1, 0))
	srv := httptest.NewServer(n.Handler())
	defer srv.Close()
	c := NewHTTPClient("n7", srv.URL)
	for _, r := range prompts(5) {
		c.Infer(context.Background(), r)
	}

	s, err := c.Stats(context.Background())
	if err != nil || s.Requests != 5 || s.Tokens != 80 || s.Errors != 0 || s.LatencyMeanMs <= 0 {
		t.Fatalf("stats: %+v err=%v", s, err)
	}

	get := func(path string) (int, string) {
		res, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		b, _ := io.ReadAll(res.Body)
		return res.StatusCode, string(b)
	}
	if code, body := get("/healthz"); code != 200 || !strings.Contains(body, `"node":"n7"`) {
		t.Fatalf("healthz: %d %s", code, body)
	}
	_, prom := get("/metrics")
	for _, want := range []string{"kllm_node_requests_total 5", "kllm_node_tokens_total 80", "kllm_node_latency_seconds_count 5"} {
		if !strings.Contains(prom, want) {
			t.Errorf("/metrics missing %q", want)
		}
	}

	res, _ := http.Post(srv.URL+"/v1/infer", "application/json", strings.NewReader(`{"prompt":[]}`))
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty prompt: status %d", res.StatusCode)
	}
	res.Body.Close()
}

// A node can front another node: the engine (or here, a replica) sits
// upstream and the fault profile is applied on the way out.
func TestRemoteBackendChainsNodes(t *testing.T) {
	m := DefaultSimModel(1)
	upstream := serve(t, New(Config{ID: "engine"}, NewSim(m, 1, 0)))
	f, _ := Preset("collude")
	front := New(Config{ID: "front", Labels: true}, Inject(Remote{C: upstream}, f, 9, m.Vocab))

	r := prompts(1)[0]
	ref, _ := m.Reference(r.Prompt, r.MaxNewTokens)
	if got := upstream.Infer(context.Background(), r); !cluster.Equal(got.Tokens, ref) {
		t.Fatal("upstream should be honest")
	}
	got := front.Infer(context.Background(), r)
	if !got.OK() || cluster.Equal(got.Tokens, ref) || !got.Label.Attacked {
		t.Fatalf("fronted node should tamper: %+v", got)
	}

	gone := New(Config{ID: "front"}, Remote{C: NewHTTPClient("gone", "http://127.0.0.1:1")})
	if got := gone.Infer(context.Background(), r); got.OK() {
		t.Fatal("an unreachable upstream must surface as an error")
	}
}
