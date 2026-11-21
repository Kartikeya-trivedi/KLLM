package cluster

import "math"

// Key returns a string that is equal for two token sequences iff the
// sequences are equal — the bucket key for sequence-level voting.
func Key(tokens []int32) string {
	b := make([]byte, 4*len(tokens))
	for i, t := range tokens {
		b[4*i] = byte(t)
		b[4*i+1] = byte(t >> 8)
		b[4*i+2] = byte(t >> 16)
		b[4*i+3] = byte(t >> 24)
	}
	return string(b)
}

// Equal reports whether two token sequences are identical.
func Equal(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// LCP is the length of the longest common prefix of a and b. For
// autoregressive output this is the natural agreement measure: once two
// decodes diverge they are conditioning on different contexts.
func LCP(a, b []int32) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// EditDistance is the Levenshtein distance between a and b in tokens.
func EditDistance(a, b []int32) int {
	if len(a) < len(b) {
		a, b = b, a
	}
	if len(b) == 0 {
		return len(a)
	}
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

// Similarity maps edit distance into [0,1]: 1 for identical sequences, 0 for
// sequences with nothing in common. Two empty sequences are identical.
func Similarity(a, b []int32) float64 {
	n := max(len(a), len(b))
	if n == 0 {
		return 1
	}
	return 1 - float64(EditDistance(a, b))/float64(n)
}

// SeqConf summarises per-token confidences as their geometric mean — the
// per-token-normalised sequence probability. ok is false when the node
// reported no confidences.
func SeqConf(conf []float32) (p float64, ok bool) {
	if len(conf) == 0 {
		return 0, false
	}
	var sum float64
	for _, c := range conf {
		sum += math.Log(math.Min(1, math.Max(1e-6, float64(c))))
	}
	return math.Exp(sum / float64(len(conf))), true
}
