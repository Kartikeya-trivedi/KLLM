// bench measures TTFT (prefill latency) and ITL (per-decode-token latency)
// for the engine, with fused kernels togglable for honest A/B comparison.
package main

import (
	"encoding/json"
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
	asJSON := flag.Bool("json", false, "emit one JSON result object (for tools/wandb_bench.py)")
	flag.Parse()

	if err := run(*dllPath, *modelDir, *device, *promptLen, *steps, *reps, *fused, *maxSeq, *asJSON); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// Result is the machine-readable benchmark record (--json), consumed by the
// W&B logger to track tok/s across kernel-optimization iterations.
type Result struct {
	Model      string  `json:"model"`
	Fused      bool    `json:"fused"`
	PromptLen  int     `json:"prompt_len"`
	Steps      int     `json:"steps"`
	Reps       int     `json:"reps"`
	TTFTms     float64 `json:"ttft_ms"`
	ITLms      float64 `json:"itl_ms"`
	DecodeTPS  float64 `json:"decode_tok_s"`
}

func run(dllPath, modelDir string, device, promptLen, steps, reps int, fused bool, maxSeq int64, asJSON bool) error {
	e, err := engine.New(dllPath, modelDir, engine.Options{Device: device, MaxSeq: maxSeq})
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
			seq := e.NewSequence()
			logits, err = seq.Forward(prompt)
			seq.Release()
			if err != nil {
				return err
			}
		}
		ttft := time.Since(t0) / prefills

		// One more prefill so decode timing starts from a clean sequence.
		seq := e.NewSequence()
		logits, err = seq.Forward(prompt)
		if err != nil {
			return err
		}
		t1 := time.Now()
		for range steps {
			next := engine.Argmax(logits)
			logits, err = seq.Forward([]int32{next})
			if err != nil {
				return err
			}
		}
		itl := time.Since(t1) / time.Duration(steps)
		seq.Release()

		if r == 0 || ttft < bestTTFT {
			bestTTFT = ttft
		}
		if r == 0 || itl < bestITL {
			bestITL = itl
		}
	}

	res := Result{
		Model:     modelDir,
		Fused:     fused,
		PromptLen: promptLen,
		Steps:     steps,
		Reps:      reps,
		TTFTms:    float64(bestTTFT) / float64(time.Millisecond),
		ITLms:     float64(bestITL) / float64(time.Millisecond),
		DecodeTPS: float64(time.Second) / float64(bestITL),
	}
	if asJSON {
		return json.NewEncoder(os.Stdout).Encode(res)
	}
	fmt.Printf("fused=%v prompt=%d steps=%d (best of %d reps, aggregate-timed)\n", fused, promptLen, steps, reps)
	fmt.Printf("TTFT (prefill %d tok)  %v\n", promptLen, bestTTFT)
	fmt.Printf("ITL avg               %v\n", bestITL)
	fmt.Printf("decode                %.1f tok/s\n", res.DecodeTPS)
	return nil
}
