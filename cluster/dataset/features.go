package dataset

import (
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
	"strconv"
	"time"

	"kllm/cluster"
	"kllm/cluster/record"
)

// Row is one node response as a detector sees it: features computed only
// from what the orchestrator can observe (this answer against the others
// on the same request), followed by the labels it is scored against.
type Row struct {
	Scenario  string `csv:"scenario"`
	RequestID string `csv:"request_id"`
	Seq       int    `csv:"seq"`
	Node      string `csv:"node"`

	// The response itself.
	OK        bool    `csv:"ok"`
	Len       int     `csv:"len"`
	MeanConf  float64 `csv:"mean_conf"` // geometric mean of reported token probabilities
	MinConf   float64 `csv:"min_conf"`
	LatencyMs float64 `csv:"latency_ms"`
	LatencyZ  float64 `csv:"latency_z"` // robust z-score among this request's responses

	// The response against its peers.
	MeanSim      float64 `csv:"mean_sim"`      // mean edit similarity to the other answers
	AgreeFrac    float64 `csv:"agree_frac"`    // fraction of other answers identical to this one
	InPlurality  bool    `csv:"in_plurality"`  // this is the most common answer
	PluralityLCP float64 `csv:"plurality_lcp"` // shared prefix with the most common answer, as a fraction

	// The request as a whole.
	Answers  int     `csv:"answers"`   // nodes that answered
	Distinct int     `csv:"distinct"`  // distinct answers among them
	TopShare float64 `csv:"top_share"` // fraction giving the most common answer
	Entropy  float64 `csv:"entropy"`   // bits, of the answer distribution

	// Labels — never inputs.
	Fault    string  `csv:"fault"`
	Attacked bool    `csv:"attacked"`
	Correct  bool    `csv:"correct"` // exact match with the reference
	RefSim   float64 `csv:"ref_sim"` // edit similarity to the reference
}

// Featurize flattens a request log into one Row per node response.
func Featurize(recs []record.Record) []Row {
	var rows []Row
	for _, rec := range recs {
		answered := cluster.Answered(rec.Responses)
		counts := map[string]int{}
		for _, r := range answered {
			counts[cluster.Key(r.Tokens)]++
		}
		var top []int32
		topKey, topCount := "", 0
		for _, r := range answered { // first-seen order makes ties deterministic
			if k := cluster.Key(r.Tokens); counts[k] > topCount {
				top, topKey, topCount = r.Tokens, k, counts[k]
			}
		}
		var entropy float64
		for _, c := range counts {
			p := float64(c) / float64(len(answered))
			entropy -= p * math.Log2(p)
		}
		latencies := make([]float64, len(rec.Responses))
		for i, r := range rec.Responses {
			latencies[i] = r.LatencyMs
		}

		for _, r := range rec.Responses {
			row := Row{
				Scenario: rec.Scenario, RequestID: rec.Request.ID, Seq: rec.Seq, Node: r.Node,
				OK: r.OK(), Len: len(r.Tokens), LatencyMs: r.LatencyMs,
				LatencyZ: cluster.RobustZ(r.LatencyMs, latencies),
				Answers:  len(answered), Distinct: len(counts), Entropy: entropy,
				Fault: "unknown",
			}
			if len(answered) > 0 {
				row.TopShare = float64(topCount) / float64(len(answered))
			}
			if r.Label != nil {
				row.Fault, row.Attacked = r.Label.Fault, r.Label.Attacked
			}
			if r.OK() {
				row.MeanConf, _ = cluster.SeqConf(r.Conf)
				for i, c := range r.Conf {
					if i == 0 || float64(c) < row.MinConf {
						row.MinConf = float64(c)
					}
				}
				key := cluster.Key(r.Tokens)
				if others := len(answered) - 1; others > 0 {
					var sim float64
					for _, o := range answered {
						if o.Node != r.Node {
							sim += cluster.Similarity(r.Tokens, o.Tokens)
						}
					}
					row.MeanSim = sim / float64(others)
					row.AgreeFrac = float64(counts[key]-1) / float64(others)
				}
				row.InPlurality = key == topKey
				if n := max(len(r.Tokens), len(top)); n > 0 {
					row.PluralityLCP = float64(cluster.LCP(r.Tokens, top)) / float64(n)
				}
				if rec.Reference != nil {
					row.Correct = cluster.Equal(r.Tokens, rec.Reference)
					row.RefSim = cluster.Similarity(r.Tokens, rec.Reference)
				}
			}
			rows = append(rows, row)
		}
	}
	return rows
}

// Split partitions rows into train and test by request, so every response
// to one request lands on the same side (no leakage between peers).
func Split(rows []Row, testFrac float64, seed uint64) (train, test []Row) {
	for _, r := range rows {
		h := splitmix(seed ^ hashID(r.RequestID))()
		if float64(h>>11)/(1<<53) < testFrac {
			test = append(test, r)
		} else {
			train = append(train, r)
		}
	}
	return
}

func hashID(s string) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(s); i++ {
		h = (h ^ uint64(s[i])) * 0x100000001b3
	}
	return h
}

// NodeRow is one node over one sampling interval: the cumulative counters
// of two consecutive Samples turned into per-interval activity.
type NodeRow struct {
	Seq           int       `csv:"seq"`
	Time          time.Time `csv:"time"`
	Node          string    `csv:"node"`
	Requests      uint64    `csv:"requests"` // in this interval
	Errors        uint64    `csv:"errors"`
	Tokens        uint64    `csv:"tokens"`
	ErrorRate     float64   `csv:"error_rate"`
	LatencyMeanMs float64   `csv:"latency_mean_ms"` // cumulative mean, as the node reports it
	LatencyP99Ms  float64   `csv:"latency_p99_ms"`
	InFlight      int64     `csv:"in_flight"`
	HeapMB        float64   `csv:"heap_mb"`
}

// NodeSeries preprocesses a system-metrics log into per-interval rows. A
// counter that went backwards means the node restarted; that interval is
// counted from zero.
func NodeSeries(samples []record.Sample) []NodeRow {
	type last struct{ req, errs, tok uint64 }
	prev := map[string]last{}
	var rows []NodeRow
	delta := func(now, before uint64) uint64 {
		if now < before {
			return now
		}
		return now - before
	}
	for _, s := range samples {
		for _, n := range s.Nodes {
			p := prev[n.Node]
			row := NodeRow{
				Seq: s.Seq, Time: s.Time, Node: n.Node,
				Requests: delta(n.Requests, p.req), Errors: delta(n.Errors, p.errs), Tokens: delta(n.Tokens, p.tok),
				LatencyMeanMs: n.LatencyMeanMs, LatencyP99Ms: n.LatencyP99Ms, InFlight: n.InFlight, HeapMB: n.HeapMB,
			}
			if row.Requests > 0 {
				row.ErrorRate = float64(row.Errors) / float64(row.Requests)
			}
			prev[n.Node] = last{n.Requests, n.Errors, n.Tokens}
			rows = append(rows, row)
		}
	}
	return rows
}

// FaultCount summarises the responses of one fault profile.
type FaultCount struct {
	Nodes     int     `json:"nodes"`
	Responses int     `json:"responses"`
	Attacked  int     `json:"attacked"`
	Errors    int     `json:"errors"`
	Correct   float64 `json:"exact_match"` // of the responses that answered
}

// Manifest describes a generated dataset.
type Manifest struct {
	Scenario  Scenario              `json:"scenario"`
	Nodes     map[string]string     `json:"nodes"` // the answer key: node id -> fault profile
	Records   int                   `json:"records"`
	Responses int                   `json:"responses"`
	Faults    map[string]FaultCount `json:"faults"`

	AttackedFrac float64 `json:"attacked_frac"`  // responses tampered with or dropped
	Unanimous    float64 `json:"unanimous_frac"` // requests where every answer agreed
	LatencyP50Ms float64 `json:"latency_p50_ms"` // wait-for-all request latency
	LatencyP99Ms float64 `json:"latency_p99_ms"`
}

// Summarize computes a dataset's manifest from its request log.
func Summarize(sc Scenario, key map[string]string, recs []record.Record) Manifest {
	m := Manifest{Scenario: sc, Nodes: key, Records: len(recs), Faults: map[string]FaultCount{}}
	nodes := map[string]map[string]bool{}
	answered := map[string]int{}
	correct := map[string]int{}
	var attacked, unanimous int
	var latency []float64
	for _, rec := range recs {
		latency = append(latency, rec.LatencyMs)
		if share, _, n := cluster.Agreement(rec.Responses); n == len(rec.Responses) && share == 1 {
			unanimous++
		}
		for _, r := range rec.Responses {
			fault := "unknown"
			if r.Label != nil {
				fault = r.Label.Fault
			}
			c := m.Faults[fault]
			c.Responses++
			if r.Label != nil && r.Label.Attacked {
				c.Attacked++
				attacked++
			}
			if !r.OK() {
				c.Errors++
			} else {
				answered[fault]++
				if cluster.Equal(r.Tokens, rec.Reference) {
					correct[fault]++
				}
			}
			m.Faults[fault] = c
			if nodes[fault] == nil {
				nodes[fault] = map[string]bool{}
			}
			nodes[fault][r.Node] = true
			m.Responses++
		}
	}
	for fault, c := range m.Faults {
		c.Nodes = len(nodes[fault])
		if answered[fault] > 0 {
			c.Correct = float64(correct[fault]) / float64(answered[fault])
		}
		m.Faults[fault] = c
	}
	if m.Responses > 0 {
		m.AttackedFrac = float64(attacked) / float64(m.Responses)
	}
	if len(recs) > 0 {
		m.Unanimous = float64(unanimous) / float64(len(recs))
	}
	m.LatencyP50Ms, m.LatencyP99Ms = cluster.Percentile(latency, 0.5), cluster.Percentile(latency, 0.99)
	return m
}

// WriteCSV writes a slice of structs as CSV, one column per field tagged
// `csv:"name"`, in declaration order, with a header row.
func WriteCSV(w io.Writer, rows any) error {
	v := reflect.ValueOf(rows)
	if v.Kind() != reflect.Slice || v.Type().Elem().Kind() != reflect.Struct {
		return fmt.Errorf("dataset: WriteCSV wants a slice of structs, got %T", rows)
	}
	t := v.Type().Elem()
	var cols []int
	var header []string
	for i := 0; i < t.NumField(); i++ {
		if name := t.Field(i).Tag.Get("csv"); name != "" {
			cols, header = append(cols, i), append(header, name)
		}
	}
	cw := csv.NewWriter(w)
	if err := cw.Write(header); err != nil {
		return err
	}
	line := make([]string, len(cols))
	for r := 0; r < v.Len(); r++ {
		for c, i := range cols {
			line[c] = cell(v.Index(r).Field(i))
		}
		if err := cw.Write(line); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}

func cell(f reflect.Value) string {
	switch x := f.Interface().(type) {
	case string:
		return x
	case bool:
		if x {
			return "1"
		}
		return "0"
	case float64:
		return strconv.FormatFloat(x, 'g', 6, 64)
	case time.Time:
		return x.UTC().Format(time.RFC3339Nano)
	}
	return fmt.Sprint(f.Interface())
}

// SortedFaults lists a manifest's fault profiles, honest first.
func (m Manifest) SortedFaults() []string {
	names := make([]string, 0, len(m.Faults))
	for f := range m.Faults {
		names = append(names, f)
	}
	sort.Slice(names, func(i, j int) bool {
		if (names[i] == "none") != (names[j] == "none") {
			return names[i] == "none"
		}
		return names[i] < names[j]
	})
	return names
}
