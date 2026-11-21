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
	"kllm/cluster/node"
	"kllm/cluster/orchestrator"
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
	flag.Parse()

	opts := orchestrator.Options{
		Policy: cluster.Policy{
			Mode: *mode, K: *k, Initial: *initial, Step: *step, Order: *order,
			TimeoutMs: float64(*timeout) / float64(time.Millisecond),
		},
		HealthEvery: *health,
	}
	if err := run(*addr, *nodes, opts); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(addr, nodes string, opts orchestrator.Options) error {
	switch opts.Policy.Mode {
	case cluster.ModeAll, cluster.ModeFirstK, cluster.ModeAdaptive:
	default:
		return fmt.Errorf("unknown policy %q", opts.Policy.Mode)
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
