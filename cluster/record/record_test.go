package record

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"kllm/cluster"
	"kllm/cluster/node"
	"kllm/cluster/orchestrator"
)

func sampleRecord(seq int) Record {
	return Record{
		Time: time.Date(2026, 1, 1, 0, 0, seq, 0, time.UTC), Scenario: "collude", Seq: seq,
		Request:   cluster.Request{ID: "r", Prompt: []int32{1, 2}, MaxNewTokens: 2},
		Reference: []int32{5, 6},
		Responses: []cluster.Response{
			{Node: "a", Tokens: []int32{5, 6}, Conf: []float32{0.9, 0.8}, LatencyMs: 101.5,
				Label: &cluster.Label{Fault: "none"}},
			{Node: "b", Err: "node crashed (injected fault)", LatencyMs: 2,
				Label: &cluster.Label{Fault: "crash", Attacked: true}},
		},
		Decision: &cluster.Decision{Tokens: []int32{5, 6}, Winners: []string{"a"}, Share: 1, Margin: 1},
		Policy:   "all", LatencyMs: 101.5, Queried: 2, Rounds: 1,
	}
}

func TestRoundTripThroughAFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "run", "requests.jsonl")
	w, err := Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := w.Write(sampleRecord(i)); err != nil {
			t.Fatal(err)
		}
	}
	if w.Count() != 3 {
		t.Fatalf("count = %d", w.Count())
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := Read(path)
	if err != nil || len(got) != 3 {
		t.Fatalf("read %d records, err=%v", len(got), err)
	}
	r, want := got[2], sampleRecord(2)
	if !r.Time.Equal(want.Time) || r.Seq != 2 || r.Scenario != "collude" {
		t.Fatalf("header fields: %+v", r)
	}
	if !cluster.Equal(r.Reference, want.Reference) || len(r.Responses) != 2 {
		t.Fatalf("payload: %+v", r)
	}
	if l := r.Responses[1].Label; l == nil || !l.Attacked || l.Fault != "crash" || r.Responses[1].OK() {
		t.Fatalf("labels and errors must survive the log: %+v", r.Responses[1])
	}
	if r.Decision == nil || r.Decision.Share != 1 || r.Responses[0].Conf[1] != 0.8 {
		t.Fatalf("decision/conf: %+v", r)
	}
}

func TestScanSkipsBlankLinesAndRejectsGarbage(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.Write(sampleRecord(0))
	w.Write(sampleRecord(1))
	w.Flush()

	n := 0
	padded := "\n" + strings.ReplaceAll(buf.String(), "\n", "\r\n\n")
	if err := Scan(strings.NewReader(padded), func(Record) error { n++; return nil }); err != nil || n != 2 {
		t.Fatalf("scanned %d, err=%v", n, err)
	}
	// No trailing newline on the last line is still a line.
	n = 0
	Scan(strings.NewReader(strings.TrimRight(buf.String(), "\n")), func(Record) error { n++; return nil })
	if n != 2 {
		t.Fatalf("unterminated last line dropped: %d", n)
	}

	err := Scan(strings.NewReader(buf.String()+"{not json\n"), func(Record) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "line 3") {
		t.Fatalf("malformed line: err = %v", err)
	}
}

func TestWriterIsConcurrencySafe(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				w.Write(sampleRecord(i))
			}
		}()
	}
	wg.Wait()
	w.Flush()
	n := 0
	if err := Scan(&buf, func(Record) error { n++; return nil }); err != nil || n != 400 {
		t.Fatalf("interleaved writes corrupted the log: n=%d err=%v", n, err)
	}
}

// End to end: an orchestrator logging through a Writer, and the sampler
// recording node counters alongside it.
func TestOrchestratorSinkAndSampler(t *testing.T) {
	dir := t.TempDir()
	reqLog, _ := Create(filepath.Join(dir, "requests.jsonl"))
	metLog, _ := Create(filepath.Join(dir, "metrics.jsonl"))

	seq := 0
	o := orchestrator.New(orchestrator.Options{Sink: func(res orchestrator.Result) {
		reqLog.Write(FromResult(res, seq, time.Now()))
		seq++
	}})
	m := node.DefaultSimModel(1)
	for _, id := range []string{"a", "b", "c"} {
		o.Add(node.New(node.Config{ID: id, Labels: true}, node.NewSim(m, node.SeedFor(id), 0)))
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { RunSampler(ctx, 5*time.Millisecond, o.Stats, metLog); close(done) }()
	for i := 0; i < 20; i++ {
		if _, err := o.Infer(ctx, cluster.Request{Prompt: []int32{int32(i), 7, 9}, MaxNewTokens: 8}); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(15 * time.Millisecond)
	cancel()
	<-done
	reqLog.Close()
	metLog.Close()

	recs, err := Read(filepath.Join(dir, "requests.jsonl"))
	if err != nil || len(recs) != 20 {
		t.Fatalf("request log: %d records, err=%v", len(recs), err)
	}
	for i, r := range recs {
		if r.Seq != i || len(r.Responses) != 3 || r.Queried != 3 || r.Request.ID == "" || r.Time.IsZero() {
			t.Fatalf("record %d: %+v", i, r)
		}
	}
	samples, err := ReadSamples(filepath.Join(dir, "metrics.jsonl"))
	if err != nil || len(samples) < 2 {
		t.Fatalf("metrics log: %d samples, err=%v", len(samples), err)
	}
	last := samples[len(samples)-1]
	if len(last.Nodes) != 3 || last.Nodes[0].Node != "a" || last.Nodes[0].Requests != 20 || last.Nodes[0].Tokens != 160 {
		t.Fatalf("final sample: %+v", last)
	}
}
