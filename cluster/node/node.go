package node

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"runtime"
	"sync/atomic"
	"time"

	"kllm/cluster"
	"kllm/engine/metrics"
)

// ErrOverloaded is returned when a node is at its concurrency limit.
var ErrOverloaded = errors.New("node overloaded")

// Config is a node's identity and serving behaviour.
type Config struct {
	ID string

	// Realtime makes the node actually wait out a backend's modelled
	// latency, as a live cluster must. With it off the node answers
	// immediately and only reports the latency — virtual time, which lets
	// the lab generate thousands of labelled requests per second.
	Realtime bool

	// MaxConcurrent bounds in-flight requests; beyond it the node sheds
	// load with ErrOverloaded (HTTP 503). Zero means unbounded.
	MaxConcurrent int

	// Labels attaches lab ground truth to responses. Off in anything
	// resembling production: a real adversary does not announce itself.
	Labels bool
}

// Stats is a node's self-reported health and load, sampled by the
// orchestrator into the system-metrics log.
type Stats struct {
	Node          string  `json:"node"`
	Model         string  `json:"model"`
	UptimeS       float64 `json:"uptime_s"`
	Requests      uint64  `json:"requests"`
	Errors        uint64  `json:"errors"`
	Rejected      uint64  `json:"rejected"`
	InFlight      int64   `json:"in_flight"`
	Tokens        uint64  `json:"tokens"`
	LatencyMeanMs float64 `json:"latency_mean_ms"`
	LatencyP50Ms  float64 `json:"latency_p50_ms"`
	LatencyP99Ms  float64 `json:"latency_p99_ms"`
	Goroutines    int     `json:"goroutines"`
	HeapMB        float64 `json:"heap_mb"`
}

// Node serves one Backend.
type Node struct {
	cfg     Config
	backend Backend
	started time.Time
	sem     chan struct{}

	requests, errs, rejected, tokens metrics.Counter
	inflight                         atomic.Int64
	latency                          *metrics.Histogram // seconds
}

func New(cfg Config, b Backend) *Node {
	n := &Node{
		cfg: cfg, backend: b, started: time.Now(),
		latency: metrics.NewHistogram([]float64{.005, .01, .025, .05, .075, .1, .15, .2, .3, .5, 1, 2.5, 5, 10}),
	}
	if cfg.MaxConcurrent > 0 {
		n.sem = make(chan struct{}, cfg.MaxConcurrent)
	}
	return n
}

func (n *Node) ID() string { return n.cfg.ID }

// Infer runs one request. It always returns a Response; failures are
// reported in Response.Err so a caller fanning out to many nodes handles
// every outcome the same way.
func (n *Node) Infer(ctx context.Context, req cluster.Request) cluster.Response {
	resp := cluster.Response{RequestID: req.ID, Node: n.cfg.ID, Model: n.backend.Model()}
	if n.sem != nil {
		select {
		case n.sem <- struct{}{}:
			defer func() { <-n.sem }()
		default:
			n.rejected.Inc()
			resp.Err = ErrOverloaded.Error()
			return resp
		}
	}
	n.requests.Inc()
	n.inflight.Add(1)
	defer n.inflight.Add(-1)

	start := time.Now()
	out, err := n.backend.Generate(ctx, req)
	if err == nil && n.cfg.Realtime && out.Latency > 0 {
		err = sleep(ctx, out.Latency)
	}
	elapsed := time.Since(start)
	if !n.cfg.Realtime && out.Latency > 0 {
		// Virtual time: report exactly the modelled figure, so a dataset
		// does not depend on how busy the machine generating it was.
		elapsed = out.Latency
	}
	resp.LatencyMs = float64(elapsed) / float64(time.Millisecond)
	if n.cfg.Labels {
		l := out.Label
		if l.Fault == "" {
			l.Fault = FaultNone
		}
		resp.Label = &l
	}
	if err != nil {
		n.errs.Inc()
		resp.Err = err.Error()
		return resp
	}
	resp.Tokens, resp.Conf = out.Tokens, out.Conf
	n.tokens.Add(uint64(len(out.Tokens)))
	n.latency.Observe(elapsed.Seconds())
	return resp
}

func sleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Stats implements Client for an in-process node.
func (n *Node) Stats(context.Context) (Stats, error) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	_, _, mean := n.latency.Snapshot()
	return Stats{
		Node:          n.cfg.ID,
		Model:         n.backend.Model(),
		UptimeS:       time.Since(n.started).Seconds(),
		Requests:      n.requests.Value(),
		Errors:        n.errs.Value(),
		Rejected:      n.rejected.Value(),
		InFlight:      n.inflight.Load(),
		Tokens:        n.tokens.Value(),
		LatencyMeanMs: mean * 1000,
		LatencyP50Ms:  n.latency.Quantile(0.5) * 1000,
		LatencyP99Ms:  n.latency.Quantile(0.99) * 1000,
		Goroutines:    runtime.NumGoroutine(),
		HeapMB:        float64(ms.HeapAlloc) / (1 << 20),
	}, nil
}

// Handler is the node's HTTP surface:
//
//	POST /v1/infer    cluster.Request -> cluster.Response
//	GET  /healthz     liveness + identity
//	GET  /stats.json  Stats
//	GET  /metrics     Prometheus text format
func (n *Node) Handler() http.Handler {
	mux := http.NewServeMux()
	writeJSON := func(w http.ResponseWriter, status int, v any) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		json.NewEncoder(w).Encode(v)
	}

	mux.HandleFunc("POST /v1/infer", func(w http.ResponseWriter, r *http.Request) {
		var req cluster.Request
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20)).Decode(&req); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if len(req.Prompt) == 0 {
			http.Error(w, "bad request: empty prompt", http.StatusBadRequest)
			return
		}
		if req.MaxNewTokens <= 0 {
			req.MaxNewTokens = 64
		}
		resp := n.Infer(r.Context(), req)
		status := http.StatusOK
		switch {
		case resp.Err == ErrOverloaded.Error():
			status = http.StatusServiceUnavailable
		case resp.Err != "":
			status = http.StatusInternalServerError
		}
		writeJSON(w, status, resp)
	})

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "node": n.cfg.ID, "model": n.backend.Model()})
	})

	mux.HandleFunc("GET /stats.json", func(w http.ResponseWriter, r *http.Request) {
		s, _ := n.Stats(r.Context())
		writeJSON(w, http.StatusOK, s)
	})

	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		metrics.WriteCounter(w, "kllm_node_requests_total", "Inference requests accepted.", n.requests.Value())
		metrics.WriteCounter(w, "kllm_node_errors_total", "Inference requests that failed.", n.errs.Value())
		metrics.WriteCounter(w, "kllm_node_rejected_total", "Requests shed at the concurrency limit.", n.rejected.Value())
		metrics.WriteCounter(w, "kllm_node_tokens_total", "Tokens returned.", n.tokens.Value())
		metrics.WriteGauge(w, "kllm_node_in_flight", "Requests currently executing.", float64(n.inflight.Load()))
		metrics.WriteGauge(w, "kllm_node_goroutines", "Goroutines in the node process.", float64(runtime.NumGoroutine()))
		n.latency.WriteProm(w, "kllm_node_latency_seconds", "Request latency.")
	})
	return mux
}
