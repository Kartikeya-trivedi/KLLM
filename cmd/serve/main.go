// serve runs the inference server: engine + continuous-batching scheduler +
// HTTP/SSE frontend.
package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"

	"kllm/engine"
	"kllm/server"
)

func main() {
	dllPath := flag.String("backend", "build/toyengine_backend.dll", "backend shared library")
	modelDir := flag.String("model", "testmodels/tiny-llama", "checkpoint directory")
	device := flag.Int("device", 0, "CUDA device index")
	// Not 8080: Windows (WinNAT) often reserves it, causing a confusing
	// "forbidden by its access permissions" bind error on first run.
	addr := flag.String("addr", "127.0.0.1:8177", "listen address")
	maxSeq := flag.Int64("max-seq", 512, "per-sequence KV capacity")
	numBlocks := flag.Int("kv-blocks", 0, "KV pool blocks (0 = default)")
	maxBatch := flag.Int("max-batch", 16, "max sequences per forward step")
	flag.Parse()

	if err := run(*dllPath, *modelDir, *device, *addr, *maxSeq, *numBlocks, *maxBatch); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(dllPath, modelDir string, device int, addr string, maxSeq int64, numBlocks, maxBatch int) error {
	e, err := engine.New(dllPath, modelDir, engine.Options{
		Device: device, MaxSeq: maxSeq, NumBlocks: numBlocks,
	})
	if err != nil {
		return err
	}
	defer e.Close()

	sched := engine.NewScheduler(e, maxBatch)
	sched.Start()
	defer sched.Stop()

	info, _ := e.B.DeviceInfo()
	fmt.Printf("serving %s on http://%s (device: %s, max batch %d)\n",
		modelDir, addr, info, maxBatch)
	return http.ListenAndServe(addr, server.Handler(sched))
}
