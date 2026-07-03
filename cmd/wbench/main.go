// wbench microbenchmarks the decode-shaped matmul: fp32 cuBLAS vs the W4
// dequant-fused kernel, at 30B-class weight shapes.
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"kllm/engine/backend"
)

func main() {
	dllPath := flag.String("backend", "build/toyengine_backend.dll", "backend shared library")
	device := flag.Int("device", 0, "CUDA device index")
	shapes := flag.String("shapes", "4096x4096,11008x4096,4096x11008", "comma-separated MxK weight shapes")
	n := flag.Int64("n", 1, "batch rows (decode = 1)")
	iters := flag.Int64("iters", 50, "iterations per timing")
	flag.Parse()

	h, err := backend.Load(*dllPath, *device)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer h.Close()

	fmt.Printf("%-14s %-6s %-12s %-12s %-10s %-14s %-14s\n",
		"shape (MxK)", "n", "fp32 ms", "w4 ms", "speedup", "fp32 GB/s", "w4 GB/s")
	for _, s := range strings.Split(*shapes, ",") {
		parts := strings.Split(strings.TrimSpace(s), "x")
		m, _ := strconv.ParseInt(parts[0], 10, 64)
		k, _ := strconv.ParseInt(parts[1], 10, 64)
		fp32, err := h.BenchMatmul(m, k, *n, *iters, 0)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		w4, err := h.BenchMatmul(m, k, *n, *iters, 1)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		fp32Bytes := float64(m*k) * 4
		w4Bytes := float64(m*k)/2 + float64(m*k/128)*4
		fmt.Printf("%-14s %-6d %-12.3f %-12.3f %-10.2f %-14.1f %-14.1f\n",
			s, *n, fp32, w4, fp32/w4,
			fp32Bytes/(fp32*1e6), w4Bytes/(w4*1e6))
	}
}
