// generate runs greedy decode on a checkpoint with raw token ids.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"kllm/engine"
)

func main() {
	dllPath := flag.String("backend", "build/toyengine_backend.dll", "path to the CUDA backend shared library")
	modelDir := flag.String("model", "testmodels/tiny-llama", "checkpoint directory (config.json + safetensors)")
	device := flag.Int("device", 0, "CUDA device index")
	ids := flag.String("ids", "1 17 42 100 7", "space-separated prompt token ids")
	steps := flag.Int("steps", 16, "max new tokens")
	maxSeq := flag.Int64("max-seq", 256, "KV capacity in tokens")
	flag.Parse()

	if err := run(*dllPath, *modelDir, *device, *ids, *steps, *maxSeq); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(dllPath, modelDir string, device int, ids string, steps int, maxSeq int64) error {
	var prompt []int32
	for _, f := range strings.Fields(ids) {
		v, err := strconv.Atoi(f)
		if err != nil {
			return fmt.Errorf("bad token id %q", f)
		}
		prompt = append(prompt, int32(v))
	}

	e, err := engine.New(dllPath, modelDir, device, maxSeq)
	if err != nil {
		return err
	}
	defer e.Close()

	out, err := e.Generate(prompt, steps)
	if err != nil {
		return err
	}
	fmt.Printf("prompt:    %v\ngenerated: %v\n", prompt, out)
	return nil
}
