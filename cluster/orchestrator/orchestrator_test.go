package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"kllm/cluster"
	"kllm/cluster/node"
)

var model = node.DefaultSimModel(1)

var req = cluster.Request{ID: "r1", Prompt: []int32{4, 8, 15, 16, 23, 42}, MaxNewTokens: 16}

// replica builds an in-process node whose modelled latency is exactly
// 16 tokens * perToken, so timing assertions are deterministic.
func replica(id string, perToken time.Duration, realtime bool, fault string) *node.Node {
	s := node.NewSim(model, node.SeedFor(id), 0)
	s.Overhead, s.TokenTime, s.Jitter, s.TailProb = 0, perToken, 0, 0
	f, _ := node.Preset(fault)
	return node.New(node.Config{ID: id, Realtime: realtime, Labels: true},
		node.Inject(s, f, node.SeedFor(id), model.Vocab))
}

// spy is an aggregator that records what it was shown and votes by
// plurality. It also plays Truster so tests can steer dispatch order.
type spy struct {
	mu    sync.Mutex
	seen  [][]cluster.Response
	trust map[string]float64
}

func (s *spy) Name() string { return "spy" }
func (s *spy) Trust(id string) float64 {
	return s.trust[id]
}
func (s *spy) Aggregate(_ cluster.Request, resps []cluster.Response) cluster.Decision {
	s.mu.Lock()
	s.seen = append(s.seen, resps)
	s.mu.Unlock()
	share, margin, _ := cluster.Agreement(resps)
	d := cluster.Decision{Share: share, Margin: margin}
	if a := cluster.Answered(resps); len(a) > 0 {
		d.Tokens = a[0].Tokens
	}
	return d
}

func okNodes(rs []cluster.Response) []string {
	var ids []string
	for _, r := range rs {
		if r.OK() {
			ids = append(ids, r.Node)
		}
	}
	sort.Strings(ids)
	return ids
}

func TestAllModeCollectsEveryNode(t *testing.T) {
	var logged []Result
	o := New(Options{Sink: func(r Result) { logged = append(logged, r) }})
	for i, ms := range []int{2, 4, 6, 8, 10} {
		o.Add(replica(fmt.Sprintf("n%d", i), time.Duration(ms)*time.Millisecond, false, "none"))
	}
	res, err := o.Infer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := model.Reference(req.Prompt, req.MaxNewTokens)
	if len(res.Responses) != 5 || res.Queried != 5 || res.Rounds != 1 || res.Policy != "all" {
		t.Fatalf("result: %+v", res)
	}
	for _, r := range res.Responses {
		if !r.OK() || !cluster.Equal(r.Tokens, ref) || r.Label == nil {
			t.Fatalf("response from %s: %+v", r.Node, r)
		}
	}
	// Virtual time: the request "took" as long as the slowest node reported.
	if res.LatencyMs < 160 || res.LatencyMs > 170 {
		t.Fatalf("latency %.1f ms, want the slowest node's 160", res.LatencyMs)
	}
	if res.Decision != nil {
		t.Fatal("no aggregator configured, yet a decision appeared")
	}
	if len(logged) != 1 || logged[0].Request.ID != "r1" {
		t.Fatalf("sink saw %d results", len(logged))
	}

	if _, err := New(Options{}).Infer(context.Background(), req); !errors.Is(err, ErrNoNodes) {
		t.Fatalf("empty cluster: err = %v", err)
	}
	if _, err := o.Infer(context.Background(), cluster.Request{}); err == nil {
		t.Fatal("empty prompt accepted")
	}
}

func TestAggregatorSeesBlindedReceivedResponses(t *testing.T) {
	agg := &spy{}
	o := New(Options{Aggregator: agg})
	o.Add(replica("a", time.Millisecond, false, "none"))
	o.Add(replica("b", time.Millisecond, false, "collude"))
	o.Add(replica("c", time.Millisecond, false, "crash"))
	res, err := o.Infer(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if res.Decision == nil || res.Aggregator != "spy" || len(agg.seen) != 1 {
		t.Fatalf("aggregator not consulted: %+v", res)
	}
	for _, r := range agg.seen[0] {
		if r.Label != nil {
			t.Fatalf("aggregator was shown the label of %s", r.Node)
		}
	}
	if len(agg.seen[0]) != 3 { // the crash is evidence too
		t.Fatalf("aggregator saw %d responses, want all 3", len(agg.seen[0]))
	}
	labelled := 0
	for _, r := range res.Responses {
		if r.Label != nil && (r.Node == "a") != (r.Label.Fault == "none") {
			t.Fatalf("label on %s: %+v", r.Node, r.Label)
		}
		if r.Label != nil {
			labelled++
		}
	}
	if labelled != 3 {
		t.Fatal("the logged result must keep the labels")
	}
}

func TestFirstKTakesTheFastestAndCancelsTheRest(t *testing.T) {
	build := func(p cluster.Policy) *Orchestrator {
		o := New(Options{Policy: p})
		o.Add(replica("fast", 1*time.Millisecond, true, "none"))    // 16 ms
		o.Add(replica("mid", 3*time.Millisecond, true, "none"))     // 48 ms
		o.Add(replica("slow", 20*time.Millisecond, true, "none"))   // 320 ms
		o.Add(replica("slower", 25*time.Millisecond, true, "none")) // 400 ms
		return o
	}
	start := time.Now()
	res, err := build(cluster.Policy{Mode: cluster.ModeFirstK, K: 2}).Infer(context.Background(), req)
	wall := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(okNodes(res.Responses)); got != "[fast mid]" {
		t.Fatalf("quorum = %s", got)
	}
	if wall > 250*time.Millisecond || res.LatencyMs > 250 {
		t.Fatalf("first-2 waited for the stragglers: %v", wall)
	}
	cancelled := 0
	for _, r := range res.Responses {
		if r.Err == cluster.ErrCancelled {
			cancelled++
		}
	}
	if cancelled != 2 || res.Queried != 4 || res.Policy != "first-2" {
		t.Fatalf("cancelled=%d result=%+v", cancelled, res)
	}

	// The live policy and its offline replay must agree on who was used.
	full, _ := build(cluster.Policy{}).Infer(context.Background(), req)
	replayed := cluster.Policy{Mode: cluster.ModeFirstK, K: 2}.Replay(req, full.Responses, nil)
	if got := fmt.Sprint(okNodes(replayed.Used)); got != "[fast mid]" {
		t.Fatalf("replay quorum = %s", got)
	}
}

func TestAdaptiveEscalatesOnlyOnDisagreement(t *testing.T) {
	policy := cluster.Policy{Mode: cluster.ModeAdaptive, Initial: 3, Step: 2, Order: "trust"}
	build := func(liar string) *Orchestrator {
		// Trust order: a, b, c are asked first.
		agg := &spy{trust: map[string]float64{"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}}
		o := New(Options{Policy: policy, Aggregator: agg})
		for _, id := range []string{"a", "b", "c", "d", "e"} {
			fault := "none"
			if id == liar {
				fault = "collude"
			}
			o.Add(replica(id, time.Millisecond, false, fault))
		}
		return o
	}

	res, _ := build("e").Infer(context.Background(), req) // the liar is never asked
	if res.Queried != 3 || res.Rounds != 1 || fmt.Sprint(okNodes(res.Responses)) != "[a b c]" {
		t.Fatalf("agreeing first round should settle it: queried=%d rounds=%d", res.Queried, res.Rounds)
	}
	if res.Policy != "adaptive-3+2-trust" {
		t.Fatalf("policy = %q", res.Policy)
	}

	res, _ = build("b").Infer(context.Background(), req) // dissent in round one
	if res.Queried != 5 || res.Rounds != 2 {
		t.Fatalf("dissent should escalate: queried=%d rounds=%d", res.Queried, res.Rounds)
	}
}

func TestVirtualTimeDeadline(t *testing.T) {
	o := New(Options{Policy: cluster.Policy{TimeoutMs: 1000}})
	o.Add(replica("ok", time.Millisecond, false, "none"))
	o.Add(replica("hung", time.Millisecond, false, "hang"))
	res, _ := o.Infer(context.Background(), req)
	if got := fmt.Sprint(okNodes(res.Responses)); got != "[ok]" {
		t.Fatalf("answered = %s", got)
	}
	if res.LatencyMs != 1000 {
		t.Fatalf("a hung node costs the full timeout, got %.1f ms", res.LatencyMs)
	}
}

// flaky is a client whose stats probe can be switched off.
type flaky struct {
	node.Client
	down bool
}

func (f *flaky) Stats(ctx context.Context) (node.Stats, error) {
	if f.down {
		return node.Stats{}, errors.New("unreachable")
	}
	return f.Client.Stats(ctx)
}

func TestProbeTakesDeadNodesOutOfRotation(t *testing.T) {
	o := New(Options{FailAfter: 2})
	f := &flaky{Client: replica("b", time.Millisecond, false, "none")}
	o.Add(replica("a", time.Millisecond, false, "none"))
	o.Add(f)
	ctx := context.Background()

	f.down = true
	o.Probe(ctx)
	if res, _ := o.Infer(ctx, req); res.Queried != 2 {
		t.Fatal("one failed probe should not evict a node")
	}
	o.Probe(ctx)
	res, _ := o.Infer(ctx, req)
	if res.Queried != 1 || res.Responses[0].Node != "a" {
		t.Fatalf("dead node still in rotation: %+v", res.Responses)
	}
	for _, m := range o.Members() {
		if m.Healthy != (m.ID == "a") {
			t.Fatalf("membership: %+v", m)
		}
	}
	if stats := o.Stats(ctx); len(stats) != 1 || stats[0].Node != "a" {
		t.Fatalf("stats sample: %+v", stats)
	}

	f.down = false
	o.Probe(ctx)
	if res, _ := o.Infer(ctx, req); res.Queried != 2 {
		t.Fatal("recovered node not readmitted")
	}
}

func TestHTTPAPI(t *testing.T) {
	// Two real node servers, registered over the wire.
	o := New(Options{Aggregator: &spy{}})
	api := httptest.NewServer(o.Handler())
	defer api.Close()

	call := func(method, path, body string) (int, string) {
		hreq, _ := http.NewRequest(method, api.URL+path, bytes.NewBufferString(body))
		res, err := http.DefaultClient.Do(hreq)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		b, _ := io.ReadAll(res.Body)
		return res.StatusCode, string(b)
	}

	if code, _ := call("GET", "/healthz", ""); code != http.StatusServiceUnavailable {
		t.Fatalf("empty cluster healthz = %d", code)
	}
	if code, _ := call("POST", "/v1/infer", `{"prompt":[1,2,3]}`); code != http.StatusServiceUnavailable {
		t.Fatalf("infer with no nodes = %d", code)
	}
	for _, id := range []string{"n1", "n2"} {
		srv := httptest.NewServer(replica(id, time.Millisecond, false, "none").Handler())
		defer srv.Close()
		body, _ := json.Marshal(map[string]string{"id": id, "url": srv.URL})
		if code, out := call("POST", "/nodes", string(body)); code != http.StatusCreated {
			t.Fatalf("register %s: %d %s", id, code, out)
		}
	}
	if code, _ := call("POST", "/nodes", `{"id":"x","url":"ftp://nope"}`); code != http.StatusBadRequest {
		t.Fatalf("bad url accepted: %d", code)
	}

	code, body := call("POST", "/v1/infer", `{"prompt":[4,8,15,16,23,42],"max_new_tokens":16}`)
	var a Answer
	if code != 200 || json.Unmarshal([]byte(body), &a) != nil {
		t.Fatalf("infer: %d %s", code, body)
	}
	ref, _ := model.Reference(req.Prompt, 16)
	if !cluster.Equal(a.Tokens, ref) || a.Share != 1 || a.Queried != 2 || a.ID == "" {
		t.Fatalf("answer: %+v", a)
	}

	code, body = call("POST", "/v1/infer?verbose=1", `{"prompt":[4,8,15,16,23,42],"max_new_tokens":16}`)
	var full Result
	if code != 200 || json.Unmarshal([]byte(body), &full) != nil || len(full.Responses) != 2 {
		t.Fatalf("verbose infer: %d %s", code, body)
	}

	_, nodes := call("GET", "/nodes", "")
	if !strings.Contains(nodes, `"id":"n1"`) || !strings.Contains(nodes, `"healthy":true`) {
		t.Fatalf("nodes: %s", nodes)
	}
	_, prom := call("GET", "/metrics", "")
	for _, want := range []string{"kllm_cluster_requests_total 3", "kllm_cluster_node_calls_total 4", `kllm_cluster_node_healthy{node="n2"} 1`} {
		if !strings.Contains(prom, want) {
			t.Errorf("/metrics missing %q\n%s", want, prom)
		}
	}
	if code, _ := call("DELETE", "/nodes/n2", ""); code != http.StatusNoContent {
		t.Fatalf("delete: %d", code)
	}
	if code, _ := call("DELETE", "/nodes/n2", ""); code != http.StatusNotFound {
		t.Fatalf("second delete: %d", code)
	}
	if len(o.Members()) != 1 {
		t.Fatal("node not removed")
	}
}
