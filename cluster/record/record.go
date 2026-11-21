// Package record is the cluster's data-collection layer: an append-only
// JSON Lines log of every request (each node's response, the decision, the
// ground truth when known) and a second log of sampled system metrics.
//
// These two files are the interface between running a cluster and studying
// it. Everything downstream — preprocessing, aggregation comparisons,
// detector evaluation — replays them, so an experiment never needs the
// cluster a second time.
package record

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"kllm/cluster"
	"kllm/cluster/node"
	"kllm/cluster/orchestrator"
)

// Record is one request in the request log.
type Record struct {
	Time     time.Time `json:"time"`
	Scenario string    `json:"scenario,omitempty"`
	Seq      int       `json:"seq"` // arrival order within the run

	Request cluster.Request `json:"request"`

	// Reference is the ground-truth continuation, when the run has one.
	Reference []int32 `json:"reference,omitempty"`

	// Responses holds every node's answer, lab labels included. Code that
	// decides or detects must go through cluster.Blind.
	Responses []cluster.Response `json:"responses"`

	Decision   *cluster.Decision `json:"decision,omitempty"`
	Aggregator string            `json:"aggregator,omitempty"`
	Policy     string            `json:"policy"`
	LatencyMs  float64           `json:"latency_ms"`
	Queried    int               `json:"queried"`
	Rounds     int               `json:"rounds"`
}

// FromResult turns an orchestrator Result into a log record.
func FromResult(res orchestrator.Result, seq int, at time.Time) Record {
	return Record{
		Time: at.UTC(), Seq: seq, Request: res.Request, Responses: res.Responses,
		Decision: res.Decision, Aggregator: res.Aggregator, Policy: res.Policy,
		LatencyMs: res.LatencyMs, Queried: res.Queried, Rounds: res.Rounds,
	}
}

// Sample is one line of the system-metrics log: every node's self-reported
// counters at one instant. Counters are cumulative; dataset.NodeSeries
// turns consecutive samples into rates.
type Sample struct {
	Time     time.Time    `json:"time"`
	Scenario string       `json:"scenario,omitempty"`
	Seq      int          `json:"seq"` // requests completed when sampled
	Nodes    []node.Stats `json:"nodes"`
}

// Writer appends JSON values to a log, one per line. Safe for concurrent use.
type Writer struct {
	mu sync.Mutex
	bw *bufio.Writer
	c  io.Closer
	n  int
}

// NewWriter logs to w.
func NewWriter(w io.Writer) *Writer {
	lw := &Writer{bw: bufio.NewWriterSize(w, 1<<16)}
	if c, ok := w.(io.Closer); ok {
		lw.c = c
	}
	return lw
}

// Create opens (truncating) a log file, creating its directory if needed.
func Create(path string) (*Writer, error) {
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return NewWriter(f), nil
}

// Write appends one value as a line.
func (w *Writer) Write(v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.n++
	w.bw.Write(raw)
	return w.bw.WriteByte('\n')
}

// Count is the number of lines written.
func (w *Writer) Count() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.n
}

// Flush pushes buffered lines to the underlying writer, so a live log can
// be tailed.
func (w *Writer) Flush() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.bw.Flush()
}

// Close flushes and closes the log.
func (w *Writer) Close() error {
	err := w.Flush()
	if w.c != nil {
		if cerr := w.c.Close(); err == nil {
			err = cerr
		}
	}
	return err
}

// scan decodes a JSON Lines stream, calling fn per value. Blank lines are
// skipped; a malformed line is an error naming its line number, because a
// silently shortened dataset is worse than a failed load.
func scan[T any](r io.Reader, fn func(T) error) error {
	br := bufio.NewReaderSize(r, 1<<16)
	for line := 1; ; line++ {
		raw, err := br.ReadBytes('\n')
		if len(raw) > 0 {
			if trimmed := trimSpace(raw); len(trimmed) > 0 {
				var v T
				if jerr := json.Unmarshal(trimmed, &v); jerr != nil {
					return fmt.Errorf("line %d: %w", line, jerr)
				}
				if ferr := fn(v); ferr != nil {
					return ferr
				}
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func trimSpace(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r' || b[len(b)-1] == ' ') {
		b = b[:len(b)-1]
	}
	return b
}

// Scan streams a request log.
func Scan(r io.Reader, fn func(Record) error) error { return scan(r, fn) }

func readFile[T any](path string) ([]T, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []T
	if err := scan(f, func(v T) error { out = append(out, v); return nil }); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return out, nil
}

// Read loads a whole request log.
func Read(path string) ([]Record, error) { return readFile[Record](path) }

// ReadSamples loads a whole system-metrics log.
func ReadSamples(path string) ([]Sample, error) { return readFile[Sample](path) }

// RunSampler writes a Sample every interval until ctx is done, then one
// final sample. source is typically Orchestrator.Stats.
func RunSampler(ctx context.Context, every time.Duration, source func(context.Context) []node.Stats, w *Writer) {
	t := time.NewTicker(every)
	defer t.Stop()
	seq := 0
	sample := func(ctx context.Context) {
		w.Write(Sample{Time: time.Now().UTC(), Seq: seq, Nodes: source(ctx)})
		w.Flush()
		seq++
	}
	for {
		select {
		case <-ctx.Done():
			final, cancel := context.WithTimeout(context.Background(), every)
			sample(final)
			cancel()
			return
		case <-t.C:
			sample(ctx)
		}
	}
}
