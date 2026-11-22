// robusteval compares aggregation strategies and dispatch policies under
// adversarial conditions.
//
// With no --data it runs the standard sweep — every attack at every
// adversary count, generated in the lab — and writes results.csv,
// detection.csv, report.md and sweep.json. Run with no flags it reproduces
// the files checked in under bench/robustness (7 nodes, 1000 requests,
// 5 seeds):
//
//	robusteval
//
// With --data it scores an existing request log (from cmd/datagen or a
// live orchestrator's --log, provided it carries references and was
// collected under the wait-for-all policy):
//
//	robusteval --data data/collude-f2/requests.jsonl
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"kllm/cluster/dataset"
	"kllm/cluster/eval"
	"kllm/cluster/record"
)

func main() {
	nodes := flag.Int("nodes", 7, "cluster size")
	maxAdv := flag.Int("max-adversaries", 0, "largest adversary count (default: a bare majority, to show where methods break)")
	requests := flag.Int("requests", 1000, "requests per condition, per seed")
	seeds := flag.Int("seeds", 5, "independent repetitions (models, workloads, clusters) to pool")
	attacks := flag.String("attacks", "", "comma-separated attacks (default: all of "+strings.Join(dataset.Attacks, ",")+")")
	out := flag.String("out", "bench/robustness", "output directory for the sweep")
	data := flag.String("data", "", "score this request log instead of running the sweep")
	flag.Parse()

	var err error
	if *data != "" {
		err = score(*data, *nodes)
	} else {
		err = sweep(*nodes, *maxAdv, *requests, *seeds, *attacks, *out)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func sweep(nodes, maxAdv, requests, seeds int, attacks, out string) error {
	s := eval.Sweep{Nodes: nodes, MaxAdversaries: maxAdv, Requests: requests}
	for i := 1; i <= seeds; i++ {
		s.Seeds = append(s.Seeds, uint64(i))
	}
	if attacks != "" {
		s.Attacks = strings.Split(attacks, ",")
	}
	start := time.Now()
	rep, err := s.Run(context.Background(), func(done, total int, name string) {
		fmt.Fprintf(os.Stderr, "\r%3d/%d conditions  %-24s", done, total, name)
	})
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	for name, rows := range map[string]any{"results.csv": rep.Metrics, "detection.csv": rep.Detection} {
		f, err := os.Create(filepath.Join(out, name))
		if err != nil {
			return err
		}
		err = dataset.WriteCSV(f, rows)
		if cerr := f.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return err
		}
	}
	if err := os.WriteFile(filepath.Join(out, "report.md"), []byte(rep.Markdown()), 0o644); err != nil {
		return err
	}
	raw, _ := json.MarshalIndent(rep.Sweep, "", "  ")
	if err := os.WriteFile(filepath.Join(out, "sweep.json"), append(raw, '\n'), 0o644); err != nil {
		return err
	}

	fmt.Printf("%d conditions x %d methods in %s -> %s\n\n", len(rep.Detection), len(rep.Sweep.Methods),
		time.Since(start).Round(time.Millisecond), out)
	fmt.Printf("%-20s %7s", "strategy", "clean")
	for f := 1; f <= rep.Sweep.MaxAdversaries; f++ {
		fmt.Printf("  %-22s", fmt.Sprintf("worst case, f=%d", f))
	}
	fmt.Println()
	for _, rb := range rep.Robustness() {
		fmt.Printf("%-20s %6.1f%%", rb.Method, 100*rb.Clean)
		for i, w := range rb.Worst {
			fmt.Printf("  %5.1f%% %-15s", 100*w, "("+rb.Hard[i]+")")
		}
		fmt.Println()
	}
	return nil
}

func score(path string, nodes int) error {
	recs, err := record.Read(path)
	if err != nil {
		return err
	}
	if len(recs) == 0 {
		return fmt.Errorf("%s: empty log", path)
	}
	if recs[0].Reference == nil {
		return fmt.Errorf("%s: records carry no reference answer, so accuracy cannot be scored", path)
	}
	if n := len(recs[0].Responses); n > 0 {
		nodes = n
	}
	fmt.Printf("%s: %d requests, %d nodes\n\n", path, len(recs), nodes)
	fmt.Printf("%-28s %7s %7s %8s %8s %8s %8s %7s %9s\n",
		"method", "EM", "+/-", "cover", "attack", "p50 ms", "p99 ms", "nodes", "tok/ans")
	for _, m := range eval.DefaultMethods(nodes) {
		x, err := eval.Evaluate(recs, m)
		if err != nil {
			return err
		}
		fmt.Printf("%-28s %6.1f%% %6.1f%% %7.1f%% %7.1f%% %8.0f %8.0f %7.2f %9.1f\n", x.Method,
			100*x.ExactMatch, 100*x.CI95, 100*x.Coverage, 100*x.AttackSuccess, x.LatencyP50, x.LatencyP99, x.NodeCalls, x.Tokens)
	}
	fmt.Println("\n+/- is the binomial interval of this one run. It does not cover run-to-run variation, which for the")
	fmt.Println("strategies that learn is the larger effect; a sweep reports that as em_ci95 and em_seed_min / em_seed_max.")
	d := eval.Detect(recs)
	fmt.Printf("\ndetection   tampered responses: P %.1f%% R %.1f%% (stateless)  P %.1f%% R %.1f%% (with memory)\n",
		100*d.RespPrecision, 100*d.RespRecall, 100*d.SuspPrecision, 100*d.SuspRecall)
	fmt.Printf("            adversarial nodes:  P %.1f%% R %.1f%%, caught %.1f requests after first tampering, %d honest nodes suspected\n",
		100*d.NodePrecision, 100*d.NodeRecall, d.TimeToDetect, d.FalseSuspects)
	return nil
}
