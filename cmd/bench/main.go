// bench measures TTFT (prefill latency) and ITL (per-decode-token latency)
// for the engine, with fused kernels togglable for honest A/B comparison.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"kllm/engine"
)

func main() {
	dllPath := flag.String("backend", "build/toyengine_backend.dll", "backend shared library")
	modelDir := flag.String("model", "testmodels/tiny-llama", "checkpoint directory")
	device := flag.Int("device", 0, "CUDA device index")
	promptLen := flag.Int("prompt", 32, "prompt length (synthetic token ids)")
	steps := flag.Int("steps", 128, "decode steps to time")
	reps := flag.Int("reps", 3, "repetitions (best reported)")
	fused := flag.Bool("fused", true, "use fused kernels")
	maxSeq := flag.Int64("max-seq", 512, "KV capacity")
	flag.Parse()

	if err := run(*dllPath, *modelDir, *device, *promptLen, *steps, *reps, *fused, *maxSeq); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(dllPath, modelDir string, device, promptLen, steps, reps int, fused bool, maxSeq int64) error {
	e, err := engine.New(dllPath, modelDir, device, maxSeq)
	if err != nil {
		return err
	}
	defer e.Close()
	if err := e.B.SetFusion(fused); err != nil {
		return err
	}

	prompt := make([]int32, promptLen)
	for i := range prompt {
		prompt[i] = int32((i*37 + 11) % int(e.Cfg.VocabSize))
	}

	// Warmup (first launches pay one-time costs).
	if _, err := e.Generate(prompt, 8); err != nil {
		return err
	}

	// Windows' monotonic clock is too coarse (~0.5ms ticks) for per-call
	// samples, so both TTFT and ITL are measured in aggregate and divided.
	var bestTTFT, bestITL time.Duration
	for r := 0; r < reps; r++ {
		const prefills = 16
		t0 := time.Now()
		var logits []float32
		for range prefills {
			logits, err = e.Prefill(prompt)
			if err != nil {
				return err
			}
		}
		ttft := time.Since(t0) / prefills

		// One more prefill so decode timing starts from a clean sequence.
		logits, err = e.Prefill(prompt)
		if err != nil {
			return err
		}
		pos := promptLen
		t1 := time.Now()
		for range steps {
			next := engine.Argmax(logits)
			logits, err = e.B.Forward([]int32{next}, pos)
			if err != nil {
				return err
			}
			pos++
		}
		itl := time.Since(t1) / time.Duration(steps)

		if r == 0 || ttft < bestTTFT {
			bestTTFT = ttft
		}
		if r == 0 || itl < bestITL {
			bestITL = itl
		}
	}

	fmt.Printf("fused=%v prompt=%d steps=%d (best of %d reps, aggregate-timed)\n", fused, promptLen, steps, reps)
	fmt.Printf("TTFT (prefill %d tok)  %v\n", promptLen, bestTTFT)
	fmt.Printf("ITL avg               %v\n", bestITL)
	fmt.Printf("decode                %.1f tok/s\n", float64(time.Second)/float64(bestITL))
	return nil
}
