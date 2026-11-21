// node runs one inference node of the cluster.
//
// The backend is either the synthetic model (no GPU; for the lab) or an
// upstream node-protocol server — typically `kllm serve` — and either can
// have a fault profile injected in front of it:
//
//	node --id n1 --addr 127.0.0.1:9101                       # honest replica
//	node --id n2 --addr 127.0.0.1:9102 --noise 0.2           # lossier replica
//	node --id n3 --addr 127.0.0.1:9103 --fault collude       # adversary
//	node --id g0 --addr 127.0.0.1:9110 --upstream http://127.0.0.1:8177
//	                                                          # the real engine
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"kllm/cluster/node"
)

func main() {
	id := flag.String("id", "node", "node id")
	addr := flag.String("addr", "127.0.0.1:9101", "listen address")
	upstream := flag.String("upstream", "", "forward to this node-protocol server (e.g. a kllm serve URL) instead of simulating")
	modelSeed := flag.Uint64("model-seed", 1, "sim: model identity; replicas of one model share it")
	vocab := flag.Int("vocab", 512, "vocabulary size (bounds what an adversary may emit)")
	noise := flag.Float64("noise", 0.1, "sim: how lossy this replica is (0 = exact)")
	corr := flag.Float64("corr", 0, "sim: fraction of this replica's error that is common-mode")
	tokenMs := flag.Float64("token-ms", 5, "sim: modelled milliseconds per token")
	fault := flag.String("fault", "none", "fault profile: "+strings.Join(node.PresetNames(), "|")+" or a JSON object")
	realtime := flag.Bool("realtime", true, "wait out modelled latency (off = virtual time)")
	labels := flag.Bool("labels", false, "attach lab ground-truth labels to responses")
	maxConc := flag.Int("max-concurrent", 0, "shed load beyond this many in-flight requests (0 = unbounded)")
	register := flag.String("register", "", "orchestrator base URL to announce this node to")
	advertise := flag.String("advertise", "", "URL the orchestrator should reach this node at (default http://<addr>)")
	flag.Parse()

	if err := run(*id, *addr, *upstream, *modelSeed, *vocab, *noise, *corr, *tokenMs,
		*fault, *realtime, *labels, *maxConc, *register, *advertise); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(id, addr, upstream string, modelSeed uint64, vocab int, noise, corr, tokenMs float64,
	faultSpec string, realtime, labels bool, maxConc int, register, advertise string) error {
	f, err := node.ParseFault(faultSpec)
	if err != nil {
		return err
	}
	seed := node.SeedFor(id)

	var backend node.Backend
	if upstream != "" {
		backend = node.Remote{C: node.NewHTTPClient("upstream", upstream)}
	} else {
		sim := node.NewSim(node.SimModel{Seed: modelSeed, Vocab: vocab, HardFrac: 0.1}, seed, noise)
		sim.Corr = corr
		sim.TokenTime = time.Duration(tokenMs * float64(time.Millisecond))
		backend = sim
	}
	backend = node.Inject(backend, f, seed, vocab)

	n := node.New(node.Config{ID: id, Realtime: realtime, MaxConcurrent: maxConc, Labels: labels}, backend)
	srv := &http.Server{Addr: addr, Handler: n.Handler()}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
	}()
	if register != "" {
		if advertise == "" {
			advertise = "http://" + addr
		}
		go announce(ctx, register, id, advertise)
	}

	fmt.Printf("node %s serving %s on http://%s (fault: %s, realtime: %v)\n",
		id, backend.Model(), addr, f.Kind, realtime)
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}
	return nil
}

// announce registers the node with the orchestrator, retrying until it is
// accepted — nodes and orchestrator can start in any order.
func announce(ctx context.Context, orchestrator, id, url string) {
	body, _ := json.Marshal(map[string]string{"id": id, "url": url})
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
			strings.TrimRight(orchestrator, "/")+"/nodes", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if res, err := http.DefaultClient.Do(req); err == nil {
			res.Body.Close()
			if res.StatusCode/100 == 2 {
				fmt.Printf("node %s registered with %s\n", id, orchestrator)
				return
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Second):
		}
	}
}
