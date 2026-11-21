package cluster

import (
	"math"
	"testing"
)

func TestKeyDistinguishesSequences(t *testing.T) {
	seqs := [][]int32{nil, {0}, {0, 0}, {1}, {256}, {1, 0}, {0, 1}, {-1}, {65536, 3}}
	seen := map[string]int{}
	for i, s := range seqs {
		k := Key(s)
		if j, dup := seen[k]; dup {
			t.Fatalf("Key collision between %v and %v", seqs[j], s)
		}
		seen[k] = i
	}
	if Key([]int32{7, 8, 9}) != Key([]int32{7, 8, 9}) {
		t.Fatal("Key is not deterministic")
	}
}

func TestSequenceMeasures(t *testing.T) {
	cases := []struct {
		a, b []int32
		lcp  int
		dist int
		sim  float64
	}{
		{nil, nil, 0, 0, 1},
		{[]int32{1, 2, 3}, []int32{1, 2, 3}, 3, 0, 1},
		{[]int32{1, 2, 3}, []int32{1, 2, 4}, 2, 1, 2.0 / 3},
		{[]int32{1, 2, 3}, nil, 0, 3, 0},
		{[]int32{1, 2, 3, 4}, []int32{2, 3, 4}, 0, 1, 0.75},
		{[]int32{5, 6}, []int32{7, 8}, 0, 2, 0},
		{[]int32{1, 2}, []int32{1, 2, 3, 4}, 2, 2, 0.5},
	}
	for _, c := range cases {
		if got := LCP(c.a, c.b); got != c.lcp {
			t.Errorf("LCP(%v,%v) = %d, want %d", c.a, c.b, got, c.lcp)
		}
		if got := EditDistance(c.a, c.b); got != c.dist {
			t.Errorf("EditDistance(%v,%v) = %d, want %d", c.a, c.b, got, c.dist)
		}
		if got := EditDistance(c.b, c.a); got != c.dist {
			t.Errorf("EditDistance not symmetric for %v,%v", c.a, c.b)
		}
		if got := Similarity(c.a, c.b); math.Abs(got-c.sim) > 1e-12 {
			t.Errorf("Similarity(%v,%v) = %g, want %g", c.a, c.b, got, c.sim)
		}
		if Equal(c.a, c.b) != (c.dist == 0) {
			t.Errorf("Equal(%v,%v) disagrees with distance %d", c.a, c.b, c.dist)
		}
	}
}

func TestSeqConf(t *testing.T) {
	if _, ok := SeqConf(nil); ok {
		t.Fatal("empty confidences must report ok=false")
	}
	p, ok := SeqConf([]float32{0.5, 0.5, 0.5})
	if !ok || math.Abs(p-0.5) > 1e-6 {
		t.Fatalf("geometric mean of constant 0.5 = %g", p)
	}
	// Geometric, not arithmetic: one very unlikely token drags it down hard.
	p, _ = SeqConf([]float32{0.99, 0.99, 0.0001})
	if p > 0.1 {
		t.Fatalf("one near-zero token should dominate, got %g", p)
	}
	// Out-of-range reports are clamped rather than producing NaN/Inf.
	p, _ = SeqConf([]float32{0, 2, -1})
	if math.IsNaN(p) || math.IsInf(p, 0) || p <= 0 {
		t.Fatalf("clamping failed: %g", p)
	}
}

func TestBlindStripsLabelsWithoutMutating(t *testing.T) {
	in := []Response{
		{Node: "a", Tokens: []int32{1}, Label: &Label{Fault: "collude", Attacked: true}},
		{Node: "b", Err: "boom"},
	}
	out := Blind(in)
	if out[0].Label != nil {
		t.Fatal("label survived Blind")
	}
	if in[0].Label == nil {
		t.Fatal("Blind mutated its input")
	}
	if got := Answered(in); len(got) != 1 || got[0].Node != "a" {
		t.Fatalf("Answered = %+v", got)
	}
}
