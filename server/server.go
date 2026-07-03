// Package server exposes the scheduler over HTTP with SSE token streaming.
package server

import (
	"encoding/json"
	"fmt"
	"net/http"

	"kllm/engine"
)

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

func Handler(s *engine.Scheduler) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
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
