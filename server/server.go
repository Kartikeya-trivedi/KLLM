// Package server exposes the scheduler over HTTP with SSE token streaming
// plus a small browser playground at /.
package server

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"kllm/cluster"
	"kllm/engine"
)

//go:embed ui.html
var uiHTML []byte

type GenerateRequest struct {
	IDs          []int32 `json:"ids"`
	MaxNewTokens int     `json:"max_new_tokens"`
}

type tokenEvent struct {
	Token int32 `json:"token"`
}

type errorEvent struct {
	Error string `json:"error"`
}

// Identity is how this server names itself when it answers as a cluster node.
type Identity struct {
	Node  string // node id reported in /v1/infer responses
	Model string // what is being served, for logs and datasets
}

func Handler(s *engine.Scheduler) http.Handler {
	return HandlerAs(s, Identity{Node: "kllm"})
}

// HandlerAs is Handler with an explicit cluster identity.
func HandlerAs(s *engine.Scheduler, id Identity) http.Handler {
	mux := http.NewServeMux()

	// POST /v1/infer — the cluster node protocol (see kllm/cluster): one
	// JSON request, one JSON response carrying the whole continuation and
	// the probability the model put on each token. This is what makes a
	// plain `kllm serve` an (honest) node the orchestrator can fan out to.
	mux.HandleFunc("POST /v1/infer", func(w http.ResponseWriter, r *http.Request) {
		var req cluster.Request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.MaxNewTokens <= 0 {
			req.MaxNewTokens = 64
		}
		start := time.Now()
		resp := cluster.Response{RequestID: req.ID, Node: id.Node, Model: id.Model}
		status := http.StatusOK
		events, err := s.SubmitConf(req.Prompt, req.MaxNewTokens)
		if err != nil {
			resp.Err, status = err.Error(), http.StatusServiceUnavailable
		} else {
			for ev := range events {
				if ev.Err != nil {
					resp.Err, status = ev.Err.Error(), http.StatusInternalServerError
					resp.Tokens, resp.Conf = nil, nil
					break
				}
				if !ev.Done {
					resp.Tokens = append(resp.Tokens, ev.Token)
					resp.Conf = append(resp.Conf, ev.Prob)
				}
			}
		}
		resp.LatencyMs = float64(time.Since(start)) / float64(time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})

	// Prometheus scrape endpoint (Grafana-ready) + JSON snapshot (UI / W&B).
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		s.Metrics().WriteProm(w)
	})
	mux.HandleFunc("GET /stats.json", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(s.Metrics().Snapshot())
	})

	// Browser playground (embedded; no external assets).
	mux.HandleFunc("GET /", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(uiHTML)
	})

	// POST /v1/generate — SSE stream: one `data: {"token":N}` per token,
	// then `data: [DONE]`.
	mux.HandleFunc("POST /v1/generate", func(w http.ResponseWriter, r *http.Request) {
		var req GenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.MaxNewTokens <= 0 {
			req.MaxNewTokens = 64
		}
		events, err := s.Submit(req.IDs, req.MaxNewTokens)
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")

		emit := func(v any) {
			raw, _ := json.Marshal(v)
			fmt.Fprintf(w, "data: %s\n\n", raw)
			flusher.Flush()
		}
		for ev := range events {
			if ev.Err != nil {
				emit(errorEvent{Error: ev.Err.Error()})
				return
			}
			if !ev.Done {
				emit(tokenEvent{Token: ev.Token})
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	})

	return mux
}
