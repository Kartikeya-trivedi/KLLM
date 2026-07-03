// Package metrics is a tiny, dependency-free metrics layer for the engine.
// It exposes counters, gauges, and histograms, renders them in Prometheus
// text-exposition format (for /metrics + Grafana), and snapshots to JSON
// (for the browser UI and the W&B benchmark logger in tools/wandb_bench.py).
//
// Deliberately no external dependency: a serving engine's observability
// shouldn't drag in a metrics SDK, and the Prometheus text format is trivial.
package metrics

import (
	"fmt"
	"io"
	"math"
	"sync"
	"sync/atomic"
)

// Counter is a monotonically increasing integer.
type Counter struct{ v atomic.Uint64 }

func (c *Counter) Inc()            { c.v.Add(1) }
func (c *Counter) Add(n uint64)    { c.v.Add(n) }
func (c *Counter) Value() uint64   { return c.v.Load() }

// Gauge is a float that can go up or down (lock-free via bit-packing).
type Gauge struct{ bits atomic.Uint64 }

func (g *Gauge) Set(f float64)    { g.bits.Store(math.Float64bits(f)) }
func (g *Gauge) Value() float64   { return math.Float64frombits(g.bits.Load()) }

// Histogram is a cumulative histogram with fixed upper bounds.
type Histogram struct {
	mu      sync.Mutex
	bounds  []float64
	counts  []uint64 // per-bucket (non-cumulative); last is +Inf
	sum     float64
	total   uint64
}

func NewHistogram(bounds []float64) *Histogram {
	return &Histogram{bounds: bounds, counts: make([]uint64, len(bounds)+1)}
}

func (h *Histogram) Observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	i := 0
	for i < len(h.bounds) && v > h.bounds[i] {
		i++
	}
	h.counts[i]++
	h.sum += v
	h.total++
}

// Snapshot returns count, sum, and mean for JSON/UI use.
func (h *Histogram) Snapshot() (count uint64, sum, mean float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	mean = 0
	if h.total > 0 {
		mean = h.sum / float64(h.total)
	}
	return h.total, h.sum, mean
}

// Metrics is the engine's fixed set of instruments.
type Metrics struct {
	RequestsTotal Counter // requests admitted
	RequestsDone  Counter // requests finished (EOS or max tokens)
	TokensTotal   Counter // decode tokens produced
	PrefillTokens Counter // prompt tokens processed
	ForwardSteps  Counter // te_forward_batch calls

	RunningSeqs  Gauge // sequences in the current batch
	QueuedReqs   Gauge // requests waiting for a slot
	KVBlocksUsed Gauge
	KVBlocksTot  Gauge
	TokPerSec    Gauge // EWMA of decode throughput (all sequences)

	TTFT      *Histogram // time-to-first-token, seconds
	ITL       *Histogram // inter-token latency, seconds
	BatchSize *Histogram // sequences per forward step
}

func New() *Metrics {
	return &Metrics{
		TTFT:      NewHistogram([]float64{.001, .002, .005, .01, .02, .05, .1, .25, .5, 1, 2.5}),
		ITL:       NewHistogram([]float64{.0005, .001, .002, .005, .01, .02, .05, .1, .25}),
		BatchSize: NewHistogram([]float64{1, 2, 4, 8, 16, 32, 64, 128}),
	}
}

// UpdateTokPerSec folds a new (tokens, seconds) sample into the EWMA gauge.
func (m *Metrics) UpdateTokPerSec(tokens int, seconds float64) {
	if seconds <= 0 || tokens <= 0 {
		return
	}
	inst := float64(tokens) / seconds
	const alpha = 0.2
	prev := m.TokPerSec.Value()
	if prev == 0 {
		m.TokPerSec.Set(inst)
	} else {
		m.TokPerSec.Set(alpha*inst + (1-alpha)*prev)
	}
}

// WriteProm renders all metrics in Prometheus text-exposition format.
func (m *Metrics) WriteProm(w io.Writer) {
	counter := func(name, help string, v uint64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s counter\n%s %d\n", name, help, name, name, v)
	}
	gauge := func(name, help string, v float64) {
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %g\n", name, help, name, name, v)
	}
	hist := func(name, help string, h *Histogram) {
		h.mu.Lock()
		defer h.mu.Unlock()
		fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s histogram\n", name, help, name)
		var cum uint64
		for i, b := range h.bounds {
			cum += h.counts[i]
			fmt.Fprintf(w, "%s_bucket{le=\"%g\"} %d\n", name, b, cum)
		}
		cum += h.counts[len(h.bounds)]
		fmt.Fprintf(w, "%s_bucket{le=\"+Inf\"} %d\n", name, cum)
		fmt.Fprintf(w, "%s_sum %g\n%s_count %d\n", name, h.sum, name, h.total)
	}

	counter("kllm_requests_total", "Requests admitted.", m.RequestsTotal.Value())
	counter("kllm_requests_done_total", "Requests finished.", m.RequestsDone.Value())
	counter("kllm_tokens_generated_total", "Decode tokens produced.", m.TokensTotal.Value())
	counter("kllm_prefill_tokens_total", "Prompt tokens processed.", m.PrefillTokens.Value())
	counter("kllm_forward_steps_total", "Batched forward steps executed.", m.ForwardSteps.Value())
	gauge("kllm_running_sequences", "Sequences in the current batch.", m.RunningSeqs.Value())
	gauge("kllm_queued_requests", "Requests waiting for a batch slot.", m.QueuedReqs.Value())
	gauge("kllm_kv_blocks_used", "KV blocks currently allocated.", m.KVBlocksUsed.Value())
	gauge("kllm_kv_blocks_total", "KV blocks in the pool.", m.KVBlocksTot.Value())
	gauge("kllm_tokens_per_second", "Decode throughput, EWMA over all sequences.", m.TokPerSec.Value())
	hist("kllm_ttft_seconds", "Time to first token.", m.TTFT)
	hist("kllm_itl_seconds", "Inter-token latency.", m.ITL)
	hist("kllm_batch_size", "Sequences per forward step.", m.BatchSize)
}

// Snapshot is a compact view for JSON (the browser UI + W&B logger).
type Snapshot struct {
	RequestsTotal uint64  `json:"requests_total"`
	RequestsDone  uint64  `json:"requests_done"`
	TokensTotal   uint64  `json:"tokens_total"`
	PrefillTokens uint64  `json:"prefill_tokens"`
	ForwardSteps  uint64  `json:"forward_steps"`
	RunningSeqs   float64 `json:"running_sequences"`
	QueuedReqs    float64 `json:"queued_requests"`
	KVBlocksUsed  float64 `json:"kv_blocks_used"`
	KVBlocksTotal float64 `json:"kv_blocks_total"`
	TokensPerSec  float64 `json:"tokens_per_second"`
	TTFTMeanMs    float64 `json:"ttft_mean_ms"`
	ITLMeanMs     float64 `json:"itl_mean_ms"`
	AvgBatchSize  float64 `json:"avg_batch_size"`
}

func (m *Metrics) Snapshot() Snapshot {
	_, _, ttft := m.TTFT.Snapshot()
	_, _, itl := m.ITL.Snapshot()
	_, _, batch := m.BatchSize.Snapshot()
	return Snapshot{
		RequestsTotal: m.RequestsTotal.Value(),
		RequestsDone:  m.RequestsDone.Value(),
		TokensTotal:   m.TokensTotal.Value(),
		PrefillTokens: m.PrefillTokens.Value(),
		ForwardSteps:  m.ForwardSteps.Value(),
		RunningSeqs:   m.RunningSeqs.Value(),
		QueuedReqs:    m.QueuedReqs.Value(),
		KVBlocksUsed:  m.KVBlocksUsed.Value(),
		KVBlocksTotal: m.KVBlocksTot.Value(),
		TokensPerSec:  m.TokPerSec.Value(),
		TTFTMeanMs:    ttft * 1000,
		ITLMeanMs:     itl * 1000,
		AvgBatchSize:  batch,
	}
}
