package dataset

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"strings"
	"testing"

	"kllm/cluster"
	"kllm/cluster/record"
)

func generate(t *testing.T, sc Scenario, opt Options) ([]byte, []record.Record, *Lab) {
	t.Helper()
	var buf bytes.Buffer
	w := record.NewWriter(&buf)
	lab, err := Generate(context.Background(), sc, w, opt)
	if err != nil {
		t.Fatal(err)
	}
	w.Flush()
	raw := append([]byte(nil), buf.Bytes()...)
	var recs []record.Record
	if err := record.Scan(&buf, func(r record.Record) error { recs = append(recs, r); return nil }); err != nil {
		t.Fatal(err)
	}
	return raw, recs, lab
}

func standard(t *testing.T, attack string, n, f, requests int) Scenario {
	t.Helper()
	sc, err := Standard(attack, n, f, requests, 42)
	if err != nil {
		t.Fatal(err)
	}
	return sc
}

func TestNormalDataset(t *testing.T) {
	sc := standard(t, "normal", 7, 0, 400)
	_, recs, lab := generate(t, sc, Options{})
	if len(recs) != 400 || len(lab.Nodes) != 7 {
		t.Fatalf("%d records, %d nodes", len(recs), len(lab.Nodes))
	}
	for i, rec := range recs {
		if rec.Seq != i || len(rec.Responses) != 7 || len(rec.Reference) != 16 || rec.Scenario != "normal-f0" {
			t.Fatalf("record %d malformed: seq=%d responses=%d", i, rec.Seq, len(rec.Responses))
		}
		if i > 0 && !rec.Time.After(recs[i-1].Time) {
			t.Fatal("virtual clock did not advance")
		}
		for j, r := range rec.Responses {
			if r.Label == nil || r.Label.Fault != "none" || r.Label.Attacked || !r.OK() {
				t.Fatalf("honest run produced %+v", r)
			}
			if j > 0 && r.Node <= rec.Responses[j-1].Node {
				t.Fatal("responses are not in node order")
			}
		}
	}
	m := Summarize(sc, lab.Faults, recs)
	h := m.Faults["none"]
	// Mixed-quality honest replicas: imperfect, but mostly right.
	if h.Nodes != 7 || h.Correct < 0.70 || h.Correct > 0.95 || m.AttackedFrac != 0 {
		t.Fatalf("honest summary: %+v attacked=%g", h, m.AttackedFrac)
	}
	if m.Unanimous < 0.2 || m.Unanimous > 0.8 {
		t.Fatalf("unanimous fraction %.2f — replicas should disagree sometimes, not always", m.Unanimous)
	}
	if m.LatencyP50Ms < 100 || m.LatencyP99Ms <= m.LatencyP50Ms || m.LatencyP99Ms > 1000 {
		t.Fatalf("latency p50=%.0f p99=%.0f", m.LatencyP50Ms, m.LatencyP99Ms)
	}
}

func TestGenerationIsReproducible(t *testing.T) {
	sc := standard(t, "collude", 7, 2, 150)
	a, _, _ := generate(t, sc, Options{})
	b, _, _ := generate(t, sc, Options{})
	if !bytes.Equal(a, b) {
		t.Fatal("the same scenario produced two different request logs")
	}
	sc.Seed++
	if c, _, _ := generate(t, sc, Options{}); bytes.Equal(a, c) {
		t.Fatal("a different seed produced the same log")
	}
}

func TestAdversarialDatasetIsLabelled(t *testing.T) {
	sc := standard(t, "collude", 7, 2, 300)
	_, recs, lab := generate(t, sc, Options{})
	bad := 0
	for id, fault := range lab.Faults {
		if fault == "collude" {
			bad++
		} else if fault != "none" {
			t.Fatalf("node %s has fault %q", id, fault)
		}
	}
	if bad != 2 {
		t.Fatalf("answer key lists %d adversaries, want 2", bad)
	}
	for _, rec := range recs {
		var lies [][]int32
		for _, r := range rec.Responses {
			if r.Label.Fault != lab.Faults[r.Node] {
				t.Fatalf("label %q disagrees with the answer key %q", r.Label.Fault, lab.Faults[r.Node])
			}
			if r.Label.Attacked {
				lies = append(lies, r.Tokens)
			}
		}
		if len(lies) != 2 || !cluster.Equal(lies[0], lies[1]) || cluster.Equal(lies[0], rec.Reference) {
			t.Fatal("colluders should give one shared wrong answer on every request")
		}
	}
	m := Summarize(sc, lab.Faults, recs)
	if c := m.Faults["collude"]; c.Nodes != 2 || c.Attacked != 600 || c.Correct != 0 {
		t.Fatalf("collude summary: %+v", c)
	}
	if got := m.SortedFaults(); len(got) != 2 || got[0] != "none" {
		t.Fatalf("fault order: %v", got)
	}
}

func TestSleeperWakesHalfWay(t *testing.T) {
	sc := standard(t, "sleeper", 5, 2, 200)
	_, recs, _ := generate(t, sc, Options{})
	for i, rec := range recs {
		attacked := 0
		for _, r := range rec.Responses {
			if r.Label.Attacked {
				attacked++
			}
		}
		if want := map[bool]int{true: 2, false: 0}[i >= 100]; attacked != want {
			t.Fatalf("request %d: %d attacked responses, want %d", i, attacked, want)
		}
	}
}

func TestFailuresAndTimeoutsAreRecorded(t *testing.T) {
	sc := standard(t, "crash", 5, 1, 50)
	_, recs, _ := generate(t, sc, Options{})
	for _, rec := range recs {
		errs := 0
		for _, r := range rec.Responses {
			if !r.OK() {
				errs++
				if !r.Label.Attacked || len(r.Tokens) != 0 {
					t.Fatalf("crashed response: %+v", r)
				}
			}
		}
		if errs != 1 {
			t.Fatalf("%d failures, want 1", errs)
		}
	}

	hang := Scenario{Name: "hang", Seed: 1, Requests: 20, Groups: []Group{{Count: 3}, {Count: 1, Preset: "hang"}}}
	_, recs, _ = generate(t, hang, Options{})
	for _, rec := range recs {
		if rec.LatencyMs != 1000 || len(cluster.Answered(rec.Responses)) != 3 {
			t.Fatalf("hung node: latency=%g answered=%d", rec.LatencyMs, len(cluster.Answered(rec.Responses)))
		}
	}
}

// The whole wire path — JSON over loopback HTTP — must produce the same
// answers and labels as calling the nodes in-process.
func TestHTTPTransportMatchesInProcess(t *testing.T) {
	sc := standard(t, "stealth", 5, 2, 40)
	_, direct, _ := generate(t, sc, Options{})
	_, wire, _ := generate(t, sc, Options{HTTP: true})
	if len(wire) != len(direct) {
		t.Fatalf("%d vs %d records", len(wire), len(direct))
	}
	for i := range direct {
		for j, d := range direct[i].Responses {
			w := wire[i].Responses[j]
			if w.Node != d.Node || !cluster.Equal(w.Tokens, d.Tokens) || *w.Label != *d.Label || len(w.Conf) != len(d.Conf) {
				t.Fatalf("request %d node %s differs over HTTP", i, d.Node)
			}
		}
	}
}

func TestFeaturize(t *testing.T) {
	sc := standard(t, "collude", 7, 2, 300)
	_, recs, _ := generate(t, sc, Options{})
	rows := Featurize(recs)
	if len(rows) != 300*7 {
		t.Fatalf("%d rows", len(rows))
	}
	var honestSim, liarSim float64
	var honest, liars int
	for _, r := range rows {
		if !r.OK || r.Answers != 7 || r.Distinct < 2 || r.TopShare <= 0 || r.Entropy <= 0 {
			t.Fatalf("row: %+v", r)
		}
		if r.Correct != (r.RefSim == 1) || (r.Attacked && r.Correct) {
			t.Fatalf("labels inconsistent: %+v", r)
		}
		if r.Attacked {
			liarSim += r.MeanSim
			liars++
			if r.AgreeFrac < 1.0/6-1e-9 { // its one accomplice, at least
				t.Fatalf("a colluder agrees with its partner: %+v", r)
			}
		} else {
			honestSim += r.MeanSim
			honest++
		}
	}
	// What makes detection possible at all: honest answers resemble each
	// other, a fabricated one resembles only its accomplice.
	if h, l := honestSim/float64(honest), liarSim/float64(liars); h < l+0.3 {
		t.Fatalf("mean similarity honest=%.2f colluder=%.2f — no separation", h, l)
	}

	train, test := Split(rows, 0.25, 7)
	if len(train)+len(test) != len(rows) || len(test) < len(rows)/8 || len(test) > len(rows)/2 {
		t.Fatalf("split %d/%d", len(train), len(test))
	}
	side := map[string]bool{}
	for _, r := range test {
		side[r.RequestID] = true
	}
	for _, r := range train {
		if side[r.RequestID] {
			t.Fatal("a request's responses were split across train and test")
		}
	}

	var buf bytes.Buffer
	if err := WriteCSV(&buf, rows[:5]); err != nil {
		t.Fatal(err)
	}
	table, err := csv.NewReader(&buf).ReadAll()
	if err != nil || len(table) != 6 || len(table[0]) != 22 {
		t.Fatalf("csv: %d rows x %d cols, err=%v", len(table), len(table[0]), err)
	}
	if strings.Join(table[0][:5], ",") != "scenario,request_id,seq,node,ok" || table[1][0] != "collude-f2" || table[1][4] != "1" {
		t.Fatalf("csv header/row: %v / %v", table[0], table[1])
	}
	if WriteCSV(&buf, 3) == nil {
		t.Fatal("WriteCSV accepted a non-slice")
	}
}

func TestSystemMetricsAreSampledAndDifferenced(t *testing.T) {
	sc := standard(t, "crash", 4, 1, 250)
	var met bytes.Buffer
	mw := record.NewWriter(&met)
	generate(t, sc, Options{Metrics: mw, MetricsEvery: 100})
	mw.Flush()

	var samples []record.Sample
	for _, line := range strings.Split(strings.TrimSpace(met.String()), "\n") {
		var s record.Sample
		if err := json.Unmarshal([]byte(line), &s); err != nil {
			t.Fatal(err)
		}
		samples = append(samples, s)
	}
	if len(samples) != 3 || samples[0].Seq != 100 || samples[2].Seq != 250 {
		t.Fatalf("got %d samples", len(samples))
	}
	rows := NodeSeries(samples)
	if len(rows) != 12 {
		t.Fatalf("%d node rows", len(rows))
	}
	var crashed int
	for _, r := range rows {
		want := uint64(100)
		if r.Seq == 250 {
			want = 50
		}
		if r.Requests != want {
			t.Fatalf("interval ending at %d: %d requests, want %d", r.Seq, r.Requests, want)
		}
		if r.ErrorRate == 1 {
			crashed++
		} else if r.ErrorRate != 0 || r.Tokens != want*16 {
			t.Fatalf("healthy node row: %+v", r)
		}
	}
	if crashed != 3 { // the one crashed node, in each of three intervals
		t.Fatalf("crashed-node intervals: %d", crashed)
	}
}

func TestStandardValidates(t *testing.T) {
	if _, err := Standard("collude", 3, 4, 10, 1); err == nil {
		t.Fatal("more adversaries than nodes accepted")
	}
	if _, err := Standard("no-such-attack", 5, 1, 10, 1); err == nil {
		t.Fatal("unknown attack accepted")
	}
	if _, err := (Scenario{Name: "empty"}).Build(); err == nil {
		t.Fatal("a scenario with no nodes built")
	}
	for _, a := range Attacks {
		sc, err := Standard(a, 7, 2, 20, 1)
		if err != nil || sc.Nodes() != 7 {
			t.Fatalf("attack %s: nodes=%d err=%v", a, sc.Nodes(), err)
		}
	}
}
