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

	// mode 0 = fp32 cuBLAS; 1..3 = W4 kernel attempts (naive, coalesced, vectorized)
	names := []string{"fp32 cuBLAS", "w4 naive", "w4 v1 coalesced", "w4 v2 vectorized"}
	fmt.Printf("%-14s %-6s %-18s %-10s %-12s %-12s\n",
		"shape (MxK)", "n", "kernel", "ms", "eff GB/s", "vs fp32")
	for _, s := range strings.Split(*shapes, ",") {
		parts := strings.Split(strings.TrimSpace(s), "x")
		m, _ := strconv.ParseInt(parts[0], 10, 64)
		k, _ := strconv.ParseInt(parts[1], 10, 64)
		var fp32 float64
		for mode := int64(0); mode <= 3; mode++ {
			ms, err := h.BenchMatmul(m, k, *n, *iters, mode)
			if err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				os.Exit(1)
			}
			bytes := float64(m*k) * 4 // fp32 weight bytes
			if mode > 0 {
				bytes = float64(m*k)/2 + float64(m*k/128)*4 // packed + scales
			}
			speedup := "-"
			if mode == 0 {
				fp32 = ms
			} else {
				speedup = fmt.Sprintf("%.2fx", fp32/ms)
			}
			fmt.Printf("%-14s %-6d %-18s %-10.3f %-12.1f %-12s\n",
				s, *n, names[mode], ms, bytes/(ms*1e6), speedup)
		}
	}
}
