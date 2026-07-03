// Package oracle is the shared correctness harness: run a checkpoint through
// the engine and diff per-layer activations, per-step logits, and greedy
// tokens against reference dumps (gen_reference.py / numpy oracles).
package oracle

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

type RefTokens struct {
	InputIDs     []int32 `json:"input_ids"`
	GeneratedIDs []int32 `json:"generated_ids"`
}

type Manifest struct {
	NumLayers int      `json:"num_layers"`
	Prompts   []string `json:"prompts"`
}

// RepoPath resolves paths relative to the repo root from a test package two
// levels deep (engine/<pkg>).
func RepoPath(parts ...string) string {
	return filepath.Join(append([]string{"..", ".."}, parts...)...)
}

// Run loads modelDir and validates it against dumpsDir. Skips if artifacts
// are missing; fails on any divergence beyond the tolerances.
func Run(t *testing.T, modelDir, dumpsDir string, layerTol, logitsTol float64) {
	t.Helper()
	dll := RepoPath("build", "toyengine_backend.dll")
	for _, p := range []string{dll, modelDir, dumpsDir} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s — build the DLL and generate the test model + dumps first", p)
		}
	}
	var man Manifest
	raw, err := os.ReadFile(filepath.Join(dumpsDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}

	e, err := engine.New(dll, modelDir, engine.Options{MaxSeq: 256})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	L := int(e.Cfg.NumHiddenLayers)

	for i := range man.Prompts {
		pdir := filepath.Join(dumpsDir, fmt.Sprintf("prompt_%d", i))
		t.Run(fmt.Sprintf("prompt_%d", i), func(t *testing.T) {
			var ref RefTokens
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
					dbgIdx = L + 1 // HF's last entry is the final-norm output
				}
				got, err := e.B.DebugRead(dbgIdx)
				if err != nil {
					t.Fatal(err)
				}
				_, want, err := npy.ReadF32(filepath.Join(pdir, fmt.Sprintf("layer_%d.npy", hf)))
				if err != nil {
					t.Fatal(err)
				}
				if d := MaxAbsDiff(got, want); d > layerTol {
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
				if d := MaxAbsDiff(logits, refRow); d > logitsTol {
					t.Fatalf("step %d: logits diverge: max abs diff %g (tol %g)", step, d, logitsTol)
				}
				gotTok := engine.Argmax(logits)
				if gotTok != wantTok {
					t.Fatalf("step %d: token %d, reference got %d", step, gotTok, wantTok)
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

func MaxAbsDiff(a, b []float32) float64 {
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
