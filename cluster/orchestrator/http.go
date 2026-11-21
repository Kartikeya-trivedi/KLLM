package orchestrator

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"

	"kllm/cluster"
	"kllm/cluster/node"
	"kllm/engine/metrics"
)

// Answer is the client-facing reply to POST /v1/infer.
type Answer struct {
	ID        string   `json:"id"`
	Tokens    []int32  `json:"tokens"`
	Share     float64  `json:"share"`
	Margin    float64  `json:"margin"`
	Abstained bool     `json:"abstained,omitempty"`
	Suspects  []string `json:"suspects,omitempty"`
	LatencyMs float64  `json:"latency_ms"`
	Queried   int      `json:"queried"`
	Policy    string   `json:"policy"`
}

// Handler is the orchestrator's HTTP surface:
//
//	POST   /v1/infer      {prompt, max_new_tokens[, id]} -> Answer
//	                      (?verbose=1 returns the full Result: every node's response)
//	GET    /nodes         membership, health, load and trust
//	POST   /nodes         {id, url}: register a node (cmd/node --register)
//	DELETE /nodes/{id}    remove a node
//	GET    /healthz       200 while at least one node is healthy
//	GET    /stats.json    counters + membership
//	GET    /metrics       Prometheus text format
func (o *Orchestrator) Handler() http.Handler {
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
		res, err := o.Infer(r.Context(), req)
		switch {
		case errors.Is(err, ErrNoNodes):
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		case err != nil:
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if r.URL.Query().Get("verbose") != "" {
			writeJSON(w, http.StatusOK, res)
			return
		}
		a := Answer{ID: res.Request.ID, LatencyMs: res.LatencyMs, Queried: res.Queried, Policy: res.Policy}
		if d := res.Decision; d != nil {
			a.Tokens, a.Share, a.Margin, a.Abstained, a.Suspects = d.Tokens, d.Share, d.Margin, d.Abstained, d.Suspects
		}
		writeJSON(w, http.StatusOK, a)
	})

	mux.HandleFunc("GET /nodes", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, o.Members())
	})

	mux.HandleFunc("POST /nodes", func(w http.ResponseWriter, r *http.Request) {
		var in struct{ ID, URL string }
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&in); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}
		u, err := url.Parse(in.URL)
		if in.ID == "" || err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
			http.Error(w, "bad request: need id and an http(s) url", http.StatusBadRequest)
			return
		}
		o.Add(node.NewHTTPClient(in.ID, in.URL))
		writeJSON(w, http.StatusCreated, map[string]string{"id": in.ID, "url": in.URL})
	})

	mux.HandleFunc("DELETE /nodes/{id}", func(w http.ResponseWriter, r *http.Request) {
		if !o.Remove(r.PathValue("id")) {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		if len(o.healthy()) == 0 {
			http.Error(w, ErrNoNodes.Error(), http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintln(w, "ok")
	})

	mux.HandleFunc("GET /stats.json", func(w http.ResponseWriter, r *http.Request) {
		_, _, mean := o.latency.Snapshot()
		writeJSON(w, http.StatusOK, map[string]any{
			"requests":        o.requests.Value(),
			"failed":          o.failed.Value(),
			"abstained":       o.abstained.Value(),
			"node_calls":      o.nodeCalls.Value(),
			"latency_mean_ms": mean * 1000,
			"latency_p50_ms":  o.latency.Quantile(0.5) * 1000,
			"latency_p99_ms":  o.latency.Quantile(0.99) * 1000,
			"policy":          o.opts.Policy.Normalize(len(o.Members())).String(),
			"nodes":           o.Members(),
		})
	})

	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		metrics.WriteCounter(w, "kllm_cluster_requests_total", "Requests received.", o.requests.Value())
		metrics.WriteCounter(w, "kllm_cluster_failed_total", "Requests that found no healthy node.", o.failed.Value())
		metrics.WriteCounter(w, "kllm_cluster_abstained_total", "Requests the aggregator declined to answer.", o.abstained.Value())
		metrics.WriteCounter(w, "kllm_cluster_node_calls_total", "Node inferences dispatched.", o.nodeCalls.Value())
		o.latency.WriteProm(w, "kllm_cluster_latency_seconds", "End-to-end request latency.")
		members := o.Members()
		fmt.Fprintln(w, "# HELP kllm_cluster_node_healthy Whether a node is in rotation.\n# TYPE kllm_cluster_node_healthy gauge")
		for _, m := range members {
			h := 0
			if m.Healthy {
				h = 1
			}
			fmt.Fprintf(w, "kllm_cluster_node_healthy{node=%q} %d\n", m.ID, h)
		}
		fmt.Fprintln(w, "# HELP kllm_cluster_node_trust Aggregator trust in a node.\n# TYPE kllm_cluster_node_trust gauge")
		for _, m := range members {
			fmt.Fprintf(w, "kllm_cluster_node_trust{node=%q} %g\n", m.ID, m.Trust)
		}
	})
	return mux
}
