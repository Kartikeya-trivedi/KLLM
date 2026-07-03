// W4 correctness harness — separate package so it gets its own process (the
// backend allows one model per process). The oracle is HF run on the
// DEQUANTIZED fp32 checkpoint (tools/quantize_w4.py writes both): if the W4
// engine matches those dumps to fp tolerance, the dequant-fused kernel
// computes exactly dequant*matmul.
package e2ew4

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"kllm/engine"
	"kllm/engine/npy"
)

const (
	layerTol  = 2e-4
	logitsTol = 2e-3
)

type refTokens struct {
	InputIDs     []int32 `json:"input_ids"`
	GeneratedIDs []int32 `json:"generated_ids"`
}

func repoPath(parts ...string) string {
	return filepath.Join(append([]string{"..", ".."}, parts...)...)
}

func TestW4MatchesDequantReference(t *testing.T) {
	dll := repoPath("build", "toyengine_backend.dll")
	model := repoPath("testmodels", "tiny-llama-w4")
	dumps := repoPath("refdumps", "tiny-llama-w4dq")
	for _, p := range []string{dll, model, dumps} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s — run quantize_w4.py + gen_reference.py first", p)
		}
	}
	var man struct {
		NumLayers int      `json:"num_layers"`
		Prompts   []string `json:"prompts"`
	}
	raw, err := os.ReadFile(filepath.Join(dumps, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}

	e, err := engine.New(dll, model, engine.Options{MaxSeq: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	L := int(e.Cfg.NumHiddenLayers)

	for i := range man.Prompts {
		pdir := filepath.Join(dumps, fmt.Sprintf("prompt_%d", i))
		t.Run(fmt.Sprintf("prompt_%d", i), func(t *testing.T) {
			var ref refTokens
			raw, err := os.ReadFile(filepath.Join(pdir, "tokens.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &ref); err != nil {
				t.Fatal(err)
			}

			seq := e.NewSequence()
			defer seq.Release()
			if err := e.B.DebugSet(true); err != nil {
				t.Fatal(err)
			}
			logits, err := seq.Forward(ref.InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			for hf := 0; hf <= L; hf++ {
				dbgIdx := hf
				if hf == L {
					dbgIdx = L + 1
				}
				got, err := e.B.DebugRead(dbgIdx)
				if err != nil {
					t.Fatal(err)
				}
				_, want, err := npy.ReadF32(filepath.Join(pdir, fmt.Sprintf("layer_%d.npy", hf)))
				if err != nil {
					t.Fatal(err)
				}
				if d := maxAbsDiff(got, want); d > layerTol {
					t.Fatalf("layer %d diverges: max abs diff %g (tol %g)", hf, d, layerTol)
				}
			}
			if err := e.B.DebugSet(false); err != nil {
				t.Fatal(err)
			}

			lshape, refLogits, err := npy.ReadF32(filepath.Join(pdir, "logits.npy"))
			if err != nil {
				t.Fatal(err)
			}
			vocab := lshape[1]
			for step, wantTok := range ref.GeneratedIDs {
				refRow := refLogits[step*vocab : (step+1)*vocab]
				if d := maxAbsDiff(logits, refRow); d > logitsTol {
					t.Fatalf("step %d: logits diverge: max abs diff %g (tol %g)", step, d, logitsTol)
				}
				gotTok := engine.Argmax(logits)
				if gotTok != wantTok {
					t.Fatalf("step %d: token %d, dequant-HF got %d", step, gotTok, wantTok)
				}
				if step == len(ref.GeneratedIDs)-1 {
					break
				}
				logits, err = seq.Forward([]int32{gotTok})
				if err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}
