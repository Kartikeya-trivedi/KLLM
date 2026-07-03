// End-to-end correctness harness: diffs the engine against the HF oracle
// dumps (tools/gen_reference.py). Skips unless the built DLL, test model,
// and refdumps exist (see CLAUDE.md for how to produce them).
package e2e

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
	layerTol  = 1e-4 // per-layer activation max abs diff (prompt pass)
	logitsTol = 1e-3 // final logits max abs diff (per decode step)
)

type refTokens struct {
	InputIDs     []int32 `json:"input_ids"`
	GeneratedIDs []int32 `json:"generated_ids"`
}

type manifest struct {
	NumLayers int      `json:"num_layers"`
	Prompts   []string `json:"prompts"`
}

func repoPath(parts ...string) string {
	return filepath.Join(append([]string{"..", ".."}, parts...)...)
}

func TestMatchesHFReference(t *testing.T) {
	dll := repoPath("build", "toyengine_backend.dll")
	model := repoPath("testmodels", "tiny-llama")
	dumps := repoPath("refdumps", "tiny-llama")
	for _, p := range []string{dll, model, dumps} {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("missing %s — build the DLL and run make_test_model.py + gen_reference.py first", p)
		}
	}

	var man manifest
	raw, err := os.ReadFile(filepath.Join(dumps, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &man); err != nil {
		t.Fatal(err)
	}

	e, err := engine.New(dll, model, 0, 256)
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

			// --- Prefill with per-layer activation diff ---
			if err := e.B.DebugSet(true); err != nil {
				t.Fatal(err)
			}
			logits, err := e.Prefill(ref.InputIDs)
			if err != nil {
				t.Fatal(err)
			}
			// Engine debug entries: [embed, layer0.., layerL-1, final-norm] = L+2.
			// HF hidden_states dump: layer_j.npy for j=0..L where [0]=embed,
			// [j]=input to layer j, [L]=final-norm output.
			nDbg, err := e.B.DebugCount()
			if err != nil {
				t.Fatal(err)
			}
			if nDbg != L+2 {
				t.Fatalf("debug entries = %d, want %d", nDbg, L+2)
			}
			for hf := 0; hf <= L; hf++ {
				dbgIdx := hf
				if hf == L {
					dbgIdx = L + 1 // skip raw last-layer residual; HF dumps post-norm
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
					t.Fatalf("layer %d activations diverge: max abs diff %g (tol %g)", hf, d, layerTol)
				}
			}
			if err := e.B.DebugSet(false); err != nil {
				t.Fatal(err)
			}

			// --- Greedy decode: token + logits diff per step ---
			lshape, refLogits, err := npy.ReadF32(filepath.Join(pdir, "logits.npy"))
			if err != nil {
				t.Fatal(err)
			}
			vocab := lshape[1]
			pos := len(ref.InputIDs)
			for step, wantTok := range ref.GeneratedIDs {
				refRow := refLogits[step*vocab : (step+1)*vocab]
				if d := maxAbsDiff(logits, refRow); d > logitsTol {
					t.Fatalf("step %d: logits diverge: max abs diff %g (tol %g)", step, d, logitsTol)
				}
				gotTok := engine.Argmax(logits)
				if gotTok != wantTok {
					t.Fatalf("step %d: token %d, HF got %d", step, gotTok, wantTok)
				}
				if step == len(ref.GeneratedIDs)-1 {
					break
				}
				logits, err = e.B.Forward([]int32{gotTok}, pos)
				if err != nil {
					t.Fatal(err)
				}
				pos++
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
