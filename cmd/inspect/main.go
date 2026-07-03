// inspect lists the tensors in a safetensors checkpoint (file or directory).
package main

import (
	"fmt"
	"os"

	"kllm/engine/loader"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: inspect <checkpoint file or directory>")
		os.Exit(2)
	}
	m, err := loader.OpenModel(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer m.Close()

	var totalBytes, totalElems int64
	for _, t := range m.Tensors() {
		fmt.Printf("%-60s %-5s %v\n", t.Name, t.Dtype, t.Shape)
		totalBytes += t.NumBytes()
		totalElems += t.NumElements()
	}
	fmt.Printf("\n%d tensors, %d params, %.2f MiB across %d file(s)\n",
		len(m.Tensors()), totalElems, float64(totalBytes)/(1<<20), len(m.Files))
}
