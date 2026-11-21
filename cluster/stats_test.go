package cluster

import (
	"math"
	"testing"
)

func TestPercentileMedianMean(t *testing.T) {
	xs := []float64{5, 1, 3, 2, 4}
	if Median(xs) != 3 || Percentile(xs, 0) != 1 || Percentile(xs, 1) != 5 || Mean(xs) != 3 {
		t.Fatalf("median=%g p0=%g p100=%g mean=%g", Median(xs), Percentile(xs, 0), Percentile(xs, 1), Mean(xs))
	}
	if xs[0] != 5 {
		t.Fatal("Percentile sorted its input in place")
	}
	if got := Percentile([]float64{10, 20}, 0.25); got != 12.5 {
		t.Fatalf("interpolation: %g", got)
	}
	if Percentile(nil, 0.5) != 0 || Mean(nil) != 0 {
		t.Fatal("empty input must give 0")
	}
}

// The reason to use MAD: one wild value moves the standard deviation a lot
// and the MAD not at all.
func TestMADIgnoresAMinorityOfOutliers(t *testing.T) {
	clean := []float64{98, 99, 100, 101, 102}
	dirty := []float64{98, 99, 100, 101, 9000}
	if MAD(clean) != 1 || MAD(dirty) != 1 {
		t.Fatalf("MAD clean=%g dirty=%g, want 1 and 1", MAD(clean), MAD(dirty))
	}
	if z := RobustZ(9000, dirty); z < 100 {
		t.Fatalf("the outlier should score far out, got z=%g", z)
	}
	if z := RobustZ(100, dirty); z != 0 {
		t.Fatalf("the median scores 0, got %g", z)
	}
}

func TestRobustZWithNoSpread(t *testing.T) {
	same := []float64{7, 7, 7, 7, 3}
	if RobustZ(7, same) != 0 {
		t.Fatal("a value at the median of a spreadless bulk scores 0")
	}
	if z := RobustZ(3, same); z != -1e3 || math.IsInf(z, 0) {
		t.Fatalf("a departure from a spreadless bulk is capped, got %g", z)
	}
	if z := RobustZ(8, same); z != 1e3 {
		t.Fatalf("got %g", z)
	}
}
