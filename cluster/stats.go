package cluster

import (
	"math"
	"sort"
)

// Percentile returns the q-quantile (0..1) of xs by linear interpolation
// between order statistics. It does not modify xs. Empty input gives 0.
func Percentile(xs []float64, q float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	pos := q * float64(len(s)-1)
	lo := int(math.Floor(pos))
	hi := min(lo+1, len(s)-1)
	return s[lo] + (s[hi]-s[lo])*(pos-float64(lo))
}

// Median is the 0.5 quantile.
func Median(xs []float64) float64 { return Percentile(xs, 0.5) }

// MAD is the median absolute deviation from the median — a spread estimate
// that a minority of wild values cannot move, unlike the standard deviation.
func MAD(xs []float64) float64 {
	m := Median(xs)
	dev := make([]float64, len(xs))
	for i, x := range xs {
		dev[i] = math.Abs(x - m)
	}
	return Median(dev)
}

// RobustZ scores how far x sits from the bulk of xs, in units comparable to
// a standard deviation (1.4826*MAD estimates sigma for normal data). When
// the bulk has no spread at all, any departure from it is reported as
// floor-limited: +/-Inf would poison downstream arithmetic, so the score is
// capped at +/-1e3.
func RobustZ(x float64, xs []float64) float64 {
	m, mad := Median(xs), MAD(xs)
	if mad == 0 {
		switch {
		case x > m:
			return 1e3
		case x < m:
			return -1e3
		}
		return 0
	}
	return math.Max(-1e3, math.Min(1e3, (x-m)/(1.4826*mad)))
}

// Mean is the arithmetic mean; 0 for empty input.
func Mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}
