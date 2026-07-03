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
	"sync"
	"testing"

	"kllm/engine"
	"kllm/engine/backend"
	"kllm/engine/npy"
	"kllm/engine/oracle"
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

// One engine per test process: the backend allows a single model instance.
var (
	engOnce sync.Once
	eng     *engine.Engine
	engMan  manifest
	engSkip string
	engErr  error
)

func getEngine(t *testing.T) (*engine.Engine, manifest) {
	t.Helper()
	engOnce.Do(func() {
		dll := oracle.BackendLib()
		model := repoPath("testmodels", "tiny-llama")
		dumps := repoPath("refdumps", "tiny-llama")
		for _, p := range []string{dll, model, dumps} {
			if _, err := os.Stat(p); err != nil {
				engSkip = fmt.Sprintf("missing %s — build the DLL and run make_test_model.py + gen_reference.py first", p)
				return
			}
		}
		raw, err := os.ReadFile(filepath.Join(dumps, "manifest.json"))
		if err == nil {
			err = json.Unmarshal(raw, &engMan)
		}
		if err != nil {
			engErr = err
			return
		}
		eng, engErr = engine.New(dll, model, engine.Options{MaxSeq: 256})
	})
	if engSkip != "" {
		t.Skip(engSkip)
	}
	if engErr != nil {
		t.Fatal(engErr)
	}
	return eng, engMan
}

func TestMatchesHFReference(t *testing.T) {
	e, man := getEngine(t)
	dumps := repoPath("refdumps", "tiny-llama")
	L := int(e.Cfg.NumHiddenLayers)

	// Both kernel paths must match the oracle: fused (default) and unfused.
	for _, fused := range []bool{true, false} {
		if err := e.B.SetFusion(fused); err != nil {
			t.Fatal(err)
		}
		t.Run(fmt.Sprintf("fused=%v", fused), func(t *testing.T) { runPrompts(t, e, dumps, man, L) })
	}
}

func runPrompts(t *testing.T, e *engine.Engine, dumps string, man manifest, L int) {
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
			seq := e.NewSequence()
			defer seq.Release()
			if err := e.B.DebugSet(true); err != nil {
				t.Fatal(err)
			}
			logits, err := seq.Forward(ref.InputIDs)
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
				logits, err = seq.Forward([]int32{gotTok})
				if err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

// TestPagedBatchMatchesSolo runs two prompts of different lengths through
// ONE batched forward per step (paged KV, shared pool) and checks both token
// streams still match HF exactly — cross-sequence isolation + batching.
func TestPagedBatchMatchesSolo(t *testing.T) {
	e, _ := getEngine(t)
	dumps := repoPath("refdumps", "tiny-llama")

	refs := make([]refTokens, 2)
	for i := range refs {
		raw, err := os.ReadFile(filepath.Join(dumps, fmt.Sprintf("prompt_%d", i), "tokens.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &refs[i]); err != nil {
			t.Fatal(err)
		}
	}

	seqs := []*engine.Sequence{e.NewSequence(), e.NewSequence()}
	defer seqs[0].Release()
	defer seqs[1].Release()

	// Joint prefill: both prompts (different lengths) in one forward_step.
	var batch []backend.SeqForward
	for i, s := range seqs {
		sf, err := s.Step(refs[i].InputIDs)
		if err != nil {
			t.Fatal(err)
		}
		batch = append(batch, sf)
	}
	logits, err := e.B.ForwardBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range seqs {
		s.Commit(len(refs[i].InputIDs))
	}

	// Joint decode: one batched step per token; sequences retire as their
	// reference stream ends (a small preview of continuous batching).
	next := []int32{engine.Argmax(logits[0]), engine.Argmax(logits[1])}
	stepIdx := []int{0, 0}
	active := []int{0, 1}
	for len(active) > 0 {
		var stillActive []int
		batch = batch[:0]
		for _, i := range active {
			want := refs[i].GeneratedIDs[stepIdx[i]]
			if next[i] != want {
				t.Fatalf("seq %d step %d: token %d, HF got %d", i, stepIdx[i], next[i], want)
			}
			if stepIdx[i] == len(refs[i].GeneratedIDs)-1 {
				continue // sequence finished; drops out of the batch
			}
			sf, err := seqs[i].Step([]int32{next[i]})
			if err != nil {
				t.Fatal(err)
			}
			batch = append(batch, sf)
			stillActive = append(stillActive, i)
		}
		if len(stillActive) == 0 {
			break
		}
		logits, err = e.B.ForwardBatch(batch)
		if err != nil {
			t.Fatal(err)
		}
		for bi, i := range stillActive {
			seqs[i].Commit(1)
			next[i] = engine.Argmax(logits[bi])
			stepIdx[i]++
		}
		active = stillActive
	}
}

// TestSchedulerMatchesHF submits all reference prompts concurrently through
// the continuous-batching scheduler and requires every stream to reproduce
// HF's tokens exactly — correctness under mixed in-flight batches.
func TestSchedulerMatchesHF(t *testing.T) {
	e, man := getEngine(t)
	dumps := repoPath("refdumps", "tiny-llama")

	sched := engine.NewScheduler(e, 4) // smaller than prompt count: forces queueing
	sched.Start()
	defer sched.Stop()

	type reply struct {
		got []int32
		err error
	}
	replies := make([]reply, len(man.Prompts))
	refs := make([]refTokens, len(man.Prompts))
	var wg sync.WaitGroup
	for i := range man.Prompts {
		raw, err := os.ReadFile(filepath.Join(dumps, fmt.Sprintf("prompt_%d", i), "tokens.json"))
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &refs[i]); err != nil {
			t.Fatal(err)
		}
		events, err := sched.Submit(refs[i].InputIDs, len(refs[i].GeneratedIDs))
		if err != nil {
			t.Fatal(err)
		}
		wg.Add(1)
		go func(i int, events <-chan engine.GenEvent) {
			defer wg.Done()
			for ev := range events {
				if ev.Err != nil {
					replies[i].err = ev.Err
					return
				}
				if !ev.Done {
					replies[i].got = append(replies[i].got, ev.Token)
				}
			}
		}(i, events)
	}
	wg.Wait()

	for i, r := range replies {
		if r.err != nil {
			t.Fatalf("prompt %d: %v", i, r.err)
		}
		if len(r.got) != len(refs[i].GeneratedIDs) {
			t.Fatalf("prompt %d: got %d tokens, want %d", i, len(r.got), len(refs[i].GeneratedIDs))
		}
		for k := range r.got {
			if r.got[k] != refs[i].GeneratedIDs[k] {
				t.Fatalf("prompt %d step %d: token %d, HF got %d", i, k, r.got[k], refs[i].GeneratedIDs[k])
			}
		}
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
