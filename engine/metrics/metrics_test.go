package metrics

import (
	"strings"
	"testing"
)

func TestCounterGauge(t *testing.T) {
	m := New()
	m.TokensTotal.Add(5)
	m.TokensTotal.Inc()
	if m.TokensTotal.Value() != 6 {
		t.Fatalf("counter = %d, want 6", m.TokensTotal.Value())
	}
	m.TokPerSec.Set(123.5)
	if m.TokPerSec.Value() != 123.5 {
		t.Fatalf("gauge = %v, want 123.5", m.TokPerSec.Value())
	}
}

func TestHistogramBuckets(t *testing.T) {
	h := NewHistogram([]float64{1, 2, 4})
	for _, v := range []float64{0.5, 1.5, 3, 10} {
		h.Observe(v)
	}
	count, sum, mean := h.Snapshot()
	if count != 4 || sum != 15 || mean != 3.75 {
		t.Fatalf("count=%d sum=%v mean=%v, want 4/15/3.75", count, sum, mean)
	}
}

func TestEWMAThroughput(t *testing.T) {
	m := New()
	m.UpdateTokPerSec(0, 1)   // ignored
	m.UpdateTokPerSec(100, 0) // ignored (no division by zero)
	m.UpdateTokPerSec(100, 1) // first real sample seeds the EWMA
	if got := m.TokPerSec.Value(); got != 100 {
		t.Fatalf("first sample = %v, want 100", got)
	}
	m.UpdateTokPerSec(200, 1) // 0.2*200 + 0.8*100
	if got := m.TokPerSec.Value(); got < 119 || got > 121 {
		t.Fatalf("ewma = %v, want ~120", got)
	}
}

func TestPromExposition(t *testing.T) {
	m := New()
	m.TokensTotal.Add(42)
	m.RunningSeqs.Set(3)
	m.TTFT.Observe(0.01)
	var sb strings.Builder
	m.WriteProm(&sb)
	out := sb.String()
	for _, want := range []string{
		"# TYPE kllm_tokens_generated_total counter",
		"kllm_tokens_generated_total 42",
		"kllm_running_sequences 3",
		"# TYPE kllm_ttft_seconds histogram",
		"kllm_ttft_seconds_bucket{le=\"+Inf\"} 1",
		"kllm_ttft_seconds_count 1",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("exposition missing %q\n---\n%s", want, out)
		}
	}
}
