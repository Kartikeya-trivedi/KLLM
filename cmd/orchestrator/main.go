// orchestrator runs the cluster coordinator: it fans each request out to
// the registered inference nodes under a dispatch policy and returns one
// answer.
//
//	orchestrator --addr 127.0.0.1:9100 \
//	    --nodes n1=http://127.0.0.1:9101,n2=http://127.0.0.1:9102,n3=http://127.0.0.1:9103
//
// Nodes can also announce themselves (cmd/node --register http://127.0.0.1:9100).
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"kllm/cluster"
	"kllm/cluster/aggregate"
	"kllm/cluster/node"
	"kllm/cluster/orchestrator"
	"kllm/cluster/record"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:9100", "listen address")
	nodes := flag.String("nodes", "", "comma-separated id=url pairs of inference nodes")
	mode := flag.String("policy", cluster.ModeAll, "dispatch policy: all | first-k | adaptive")
	k := flag.Int("k", 0, "first-k: answers to wait for (default: a majority)")
	initial := flag.Int("initial", 3, "adaptive: nodes asked first")
	step := flag.Int("step", 2, "adaptive: nodes added per escalation")
	order := flag.String("order", "hash", "who is asked first when not everyone is: hash | trust")
	timeout := flag.Duration("timeout", 2*time.Second, "per-request deadline for node calls")
	health := flag.Duration("health-every", 2*time.Second, "node health-probe interval")
	aggName := flag.String("aggregator", "majority", "how answers are combined: none | "+strings.Join(aggregate.Names(), " | "))
	logPath := flag.String("log", "", "write every request (all node responses) to this JSONL file, one line each; an existing file is replaced")
	metricsPath := flag.String("metrics-log", "", "sample every node's stats into this JSONL file; an existing file is replaced")
	metricsEvery := flag.Duration("metrics-every", 5*time.Second, "system-metrics sampling interval")
	flag.Parse()

	opts := orchestrator.Options{
		Policy: cluster.Policy{
			Mode: *mode, K: *k, Initial: *initial, Step: *step, Order: *order,
			TimeoutMs: float64(*timeout) / float64(time.Millisecond),
		},
		HealthEvery: *health,
	}
	if *aggName != "none" { // "none" collects every response and decides nothing
		agg, err := aggregate.New(*aggName)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		opts.Aggregator = agg
	}
	if err := run(*addr, *nodes, opts, *logPath, *metricsPath, *metricsEvery); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(addr, nodes string, opts orchestrator.Options, logPath, metricsPath string, metricsEvery time.Duration) error {
	switch opts.Policy.Mode {
	case cluster.ModeAll, cluster.ModeFirstK, cluster.ModeAdaptive:
	default:
		return fmt.Errorf("unknown policy %q", opts.Policy.Mode)
	}
	if logPath != "" {
		w, err := record.Create(logPath)
		if err != nil {
			return err
		}
		defer w.Close()
		seq := 0
		opts.Sink = func(res orchestrator.Result) { // serialised by the orchestrator
			w.Write(record.FromResult(res, seq, time.Now()))
			w.Flush()
			seq++
		}
	}
	o := orchestrator.New(opts)
	for _, pair := range strings.Split(nodes, ",") {
		if pair = strings.TrimSpace(pair); pair == "" {
			continue
		}
		id, url, ok := strings.Cut(pair, "=")
		if !ok || id == "" || url == "" {
			return fmt.Errorf("bad --nodes entry %q, want id=url", pair)
		}
		o.Add(node.NewHTTPClient(id, url))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go o.Run(ctx)
	if metricsPath != "" {
		w, err := record.Create(metricsPath)
		if err != nil {
			return err
		}
		sampled := make(chan struct{})
		go func() { record.RunSampler(ctx, metricsEvery, o.Stats, w); close(sampled) }()
		defer func() { stop(); <-sampled; w.Close() }() // stop first: the sampler exits on ctx
	}

	srv := &http.Server{Addr: addr, Handler: o.Handler()}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
	}()
	fmt.Printf("orchestrator on http://%s — %d nodes, policy %s\n",
		addr, len(o.Members()), opts.Policy.Normalize(max(len(o.Members()), 1)))
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}
	return nil
}
