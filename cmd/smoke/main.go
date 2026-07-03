// smoke proves the walking skeleton: Go -> shared library -> CUDA context ->
// kernel launch -> results verified back in Go.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"

	"kllm/engine/backend"
)

func main() {
	dllPath := flag.String("backend", "build/toyengine_backend.dll", "path to the CUDA backend shared library")
	device := flag.Int("device", 0, "CUDA device index")
	n := flag.Int("n", 1<<20, "vector length for the smoke kernel")
	flag.Parse()

	if err := run(*dllPath, *device, *n); err != nil {
		fmt.Fprintln(os.Stderr, "FAIL:", err)
		os.Exit(1)
	}
}

func run(dllPath string, device, n int) error {
	h, err := backend.Load(dllPath, device)
	if err != nil {
		return err
	}
	defer h.Close()

	info, err := h.DeviceInfo()
	if err != nil {
		return err
	}
	fmt.Println("device:", info)

	rng := rand.New(rand.NewSource(42))
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = rng.Float32()
		b[i] = rng.Float32()
	}

	out, err := h.SmokeVectorAdd(a, b)
	if err != nil {
		return err
	}

	var maxErr float64
	for i := range out {
		diff := math.Abs(float64(out[i] - (a[i] + b[i])))
		if diff > maxErr {
			maxErr = diff
		}
	}
	if maxErr > 1e-6 {
		return fmt.Errorf("vector add mismatch: max abs error %g", maxErr)
	}
	fmt.Printf("smoke vector add: n=%d, max abs error %g\n", n, maxErr)
	fmt.Println("PASS")
	return nil
}
