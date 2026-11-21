// datagen generates labelled datasets from the cluster lab, under normal
// and adversarial conditions.
//
//	datagen --attack normal  --out data/normal              # all-honest baseline
//	datagen --attack collude --adversaries 2 --out data/collude-f2
//	datagen --suite --out data/suite                        # normal + every attack x adversary count
//	datagen --config my-scenario.json --out data/custom
//
// Each scenario gets a directory holding:
//
//	scenario.json     what was run (re-running it reproduces requests.jsonl exactly)
//	requests.jsonl    raw log: every node's response per request, with ground truth
//	metrics.jsonl     raw system metrics, sampled during the run
//	features.csv      one row per node response: observable features + labels
//	train.csv/test.csv  the same, split by request
//	node_metrics.csv  per-node, per-interval activity derived from metrics.jsonl
//	manifest.json     summary: answer key, per-fault counts, accuracy, latency
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"kllm/cluster/dataset"
	"kllm/cluster/record"
)

func main() {
	attack := flag.String("attack", "normal", "condition: normal | "+strings.Join(dataset.Attacks, " | "))
	nodes := flag.Int("nodes", 7, "cluster size")
	adversaries := flag.Int("adversaries", 2, "how many nodes run the attack")
	requests := flag.Int("requests", 1000, "requests per scenario")
	seed := flag.Uint64("seed", 1, "seed: fixes the model, the workload and every node's randomness")
	out := flag.String("out", "data", "output directory")
	suite := flag.Bool("suite", false, "generate normal plus every attack at 1..max adversaries")
	maxAdv := flag.Int("max-adversaries", 0, "suite: largest adversary count (default: just under half the cluster)")
	config := flag.String("config", "", "generate from a scenario JSON file instead")
	overHTTP := flag.Bool("http", false, "run nodes behind real loopback HTTP servers (slower; exercises the wire path)")
	testFrac := flag.Float64("test-frac", 0.2, "fraction of requests held out in test.csv")
	flag.Parse()

	if err := run(*attack, *nodes, *adversaries, *requests, *seed, *out, *suite, *maxAdv, *config, *overHTTP, *testFrac); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(attack string, nodes, adversaries, requests int, seed uint64, out string,
	suite bool, maxAdv int, config string, overHTTP bool, testFrac float64) error {
	var scenarios []dataset.Scenario
	switch {
	case config != "":
		raw, err := os.ReadFile(config)
		if err != nil {
			return err
		}
		var sc dataset.Scenario
		if err := json.Unmarshal(raw, &sc); err != nil {
			return fmt.Errorf("%s: %w", config, err)
		}
		if sc.Name == "" {
			sc.Name = strings.TrimSuffix(filepath.Base(config), filepath.Ext(config))
		}
		scenarios = append(scenarios, sc.WithDefaults())
	case suite:
		if maxAdv <= 0 {
			maxAdv = (nodes - 1) / 2
		}
		sc, err := dataset.Standard("normal", nodes, 0, requests, seed)
		if err != nil {
			return err
		}
		scenarios = append(scenarios, sc)
		for _, a := range dataset.Attacks {
			for f := 1; f <= maxAdv; f++ {
				if sc, err = dataset.Standard(a, nodes, f, requests, seed); err != nil {
					return err
				}
				scenarios = append(scenarios, sc)
			}
		}
	default:
		sc, err := dataset.Standard(attack, nodes, adversaries, requests, seed)
		if err != nil {
			return err
		}
		scenarios = append(scenarios, sc)
	}

	fmt.Printf("%-20s %6s %6s %9s %10s %10s %9s\n", "scenario", "nodes", "reqs", "attacked", "honest EM", "unanimous", "p99 ms")
	for _, sc := range scenarios {
		dir := out
		if len(scenarios) > 1 {
			dir = filepath.Join(out, sc.Name)
		}
		m, err := generate(sc, dir, overHTTP, testFrac)
		if err != nil {
			return err
		}
		fmt.Printf("%-20s %6d %6d %8.1f%% %9.1f%% %9.1f%% %9.0f\n", sc.Name, sc.Nodes(), m.Records,
			100*m.AttackedFrac, 100*m.Faults["none"].Correct, 100*m.Unanimous, m.LatencyP99Ms)
	}
	fmt.Println("wrote", out)
	return nil
}

func generate(sc dataset.Scenario, dir string, overHTTP bool, testFrac float64) (dataset.Manifest, error) {
	var m dataset.Manifest
	reqPath, metPath := filepath.Join(dir, "requests.jsonl"), filepath.Join(dir, "metrics.jsonl")
	reqLog, err := record.Create(reqPath)
	if err != nil {
		return m, err
	}
	metLog, err := record.Create(metPath)
	if err != nil {
		return m, err
	}
	lab, err := dataset.Generate(context.Background(), sc, reqLog, dataset.Options{HTTP: overHTTP, Metrics: metLog})
	if cerr := reqLog.Close(); err == nil {
		err = cerr
	}
	if cerr := metLog.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return m, err
	}

	// Preprocess from the files just written: the pipeline consumes the
	// same logs a live cluster would produce.
	recs, err := record.Read(reqPath)
	if err != nil {
		return m, err
	}
	samples, err := record.ReadSamples(metPath)
	if err != nil {
		return m, err
	}
	rows := dataset.Featurize(recs)
	train, test := dataset.Split(rows, testFrac, sc.Seed)
	m = dataset.Summarize(lab.Scenario, lab.Faults, recs)

	for name, table := range map[string]any{
		"features.csv": rows, "train.csv": train, "test.csv": test,
		"node_metrics.csv": dataset.NodeSeries(samples),
	} {
		f, err := os.Create(filepath.Join(dir, name))
		if err != nil {
			return m, err
		}
		err = dataset.WriteCSV(f, table)
		if cerr := f.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return m, fmt.Errorf("%s: %w", name, err)
		}
	}
	for name, v := range map[string]any{"scenario.json": lab.Scenario, "manifest.json": m} {
		raw, _ := json.MarshalIndent(v, "", "  ")
		if err := os.WriteFile(filepath.Join(dir, name), append(raw, '\n'), 0o644); err != nil {
			return m, err
		}
	}
	return m, nil
}
