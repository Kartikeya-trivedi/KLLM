package backend

import (
	"fmt"
	"path/filepath"
	"syscall"
	"unsafe"
)

// winImpl talks to toyengine_backend.dll via LoadLibrary/GetProcAddress.
// No cgo: the C-ABI is narrow enough that raw syscalls cover it, which keeps
// the Windows lab box free of the MinGW-w64 dependency.
type winImpl struct {
	dll *syscall.DLL

	procLastError      *syscall.Proc
	procInit           *syscall.Proc
	procDeviceInfo     *syscall.Proc
	procSmokeVectorAdd *syscall.Proc
	procModelCreate    *syscall.Proc
	procLoadTensor     *syscall.Proc
	procLoadTensorW4   *syscall.Proc
	procBenchMatmul    *syscall.Proc
	procLayerSliding   *syscall.Proc
	procFinalize       *syscall.Proc
	procForwardBatch   *syscall.Proc
	procSetFusion      *syscall.Proc
	procSetKernels     *syscall.Proc
	procDebugSet       *syscall.Proc
	procDebugCount     *syscall.Proc
	procDebugSize      *syscall.Proc
	procDebugRead      *syscall.Proc
}

func load(path string) (impl, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	dll, err := syscall.LoadDLL(abs)
	if err != nil {
		return nil, fmt.Errorf("loading backend dll %s: %w", abs, err)
	}
	w := &winImpl{dll: dll}
	for _, p := range []struct {
		name string
		dst  **syscall.Proc
	}{
		{"te_last_error", &w.procLastError},
		{"te_init", &w.procInit},
		{"te_device_info", &w.procDeviceInfo},
		{"te_smoke_vector_add", &w.procSmokeVectorAdd},
		{"te_model_create", &w.procModelCreate},
		{"te_model_load_tensor", &w.procLoadTensor},
		{"te_model_load_tensor_w4", &w.procLoadTensorW4},
		{"te_bench_matmul", &w.procBenchMatmul},
		{"te_model_set_layer_sliding", &w.procLayerSliding},
		{"te_model_finalize", &w.procFinalize},
		{"te_forward_batch", &w.procForwardBatch},
		{"te_set_fusion", &w.procSetFusion},
		{"te_set_kernels", &w.procSetKernels},
		{"te_debug_set", &w.procDebugSet},
		{"te_debug_count", &w.procDebugCount},
		{"te_debug_size", &w.procDebugSize},
		{"te_debug_read", &w.procDebugRead},
	} {
		proc, err := dll.FindProc(p.name)
		if err != nil {
			dll.Release()
			return nil, fmt.Errorf("backend dll missing export %s: %w", p.name, err)
		}
		*p.dst = proc
	}
	return w, nil
}

// check converts a shim return code (C `int`, so 32-bit; negative = TE_ERR_*)
// into a Go error using te_last_error.
func (w *winImpl) check(fn string, rc uintptr) error {
	code := int32(uint32(rc))
	if code == 0 {
		return nil
	}
	buf := make([]byte, 512)
	w.procLastError.Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return fmt.Errorf("%s: backend error %d: %s", fn, code, string(buf[:cstrlen(buf)]))
}

func (w *winImpl) modelCreate(cfg *ModelConfig) error {
	rc, _, _ := w.procModelCreate.Call(uintptr(unsafe.Pointer(cfg)))
	return w.check("te_model_create", rc)
}

func (w *winImpl) loadTensor(name string, f32raw []byte) error {
	cname := append([]byte(name), 0)
	rc, _, _ := w.procLoadTensor.Call(
		uintptr(unsafe.Pointer(&cname[0])),
		uintptr(unsafe.Pointer(&f32raw[0])),
		uintptr(len(f32raw)/4),
	)
	return w.check("te_model_load_tensor", rc)
}

func (w *winImpl) loadTensorW4(name string, q, scales []byte, outDim, inDim, group int64) error {
	cname := append([]byte(name), 0)
	rc, _, _ := w.procLoadTensorW4.Call(
		uintptr(unsafe.Pointer(&cname[0])),
		uintptr(unsafe.Pointer(&q[0])),
		uintptr(unsafe.Pointer(&scales[0])),
		uintptr(outDim),
		uintptr(inDim),
		uintptr(group),
	)
	return w.check("te_model_load_tensor_w4", rc)
}

func (w *winImpl) benchMatmul(m, k, n, iters, mode int64) (float64, error) {
	var ms float64
	rc, _, _ := w.procBenchMatmul.Call(
		uintptr(m), uintptr(k), uintptr(n), uintptr(iters), uintptr(mode),
		uintptr(unsafe.Pointer(&ms)),
	)
	if err := w.check("te_bench_matmul", rc); err != nil {
		return 0, err
	}
	return ms, nil
}

func (w *winImpl) setLayerSliding(flags []int32) error {
	rc, _, _ := w.procLayerSliding.Call(
		uintptr(unsafe.Pointer(&flags[0])),
		uintptr(len(flags)),
	)
	return w.check("te_model_set_layer_sliding", rc)
}

func (w *winImpl) finalize() error {
	rc, _, _ := w.procFinalize.Call()
	return w.check("te_model_finalize", rc)
}

func (w *winImpl) forwardBatch(seqs []SeqForward, logits []float32) error {
	tokens, nTokens, pos, tables, mbps := flattenBatch(seqs)

	rc, _, _ := w.procForwardBatch.Call(
		uintptr(len(seqs)),
		uintptr(unsafe.Pointer(&tokens[0])),
		uintptr(unsafe.Pointer(&nTokens[0])),
		uintptr(unsafe.Pointer(&pos[0])),
		uintptr(unsafe.Pointer(&tables[0])),
		uintptr(mbps),
		uintptr(unsafe.Pointer(&logits[0])),
	)
	return w.check("te_forward_batch", rc)
}

func (w *winImpl) setFusion(on bool) error {
	v := uintptr(0)
	if on {
		v = 1
	}
	rc, _, _ := w.procSetFusion.Call(v)
	return w.check("te_set_fusion", rc)
}

func (w *winImpl) setKernels(w4, attn int64) error {
	rc, _, _ := w.procSetKernels.Call(uintptr(w4), uintptr(attn))
	return w.check("te_set_kernels", rc)
}

func (w *winImpl) debugSet(on bool) error {
	v := uintptr(0)
	if on {
		v = 1
	}
	rc, _, _ := w.procDebugSet.Call(v)
	return w.check("te_debug_set", rc)
}

func (w *winImpl) debugCount() (int, error) {
	rc, _, _ := w.procDebugCount.Call()
	n := int64(rc)
	if n < 0 {
		return 0, fmt.Errorf("te_debug_count: %d (no model?)", n)
	}
	return int(n), nil
}

func (w *winImpl) debugRead(idx int) ([]float32, error) {
	szRc, _, _ := w.procDebugSize.Call(uintptr(idx))
	size := int64(szRc)
	if size < 0 {
		return nil, fmt.Errorf("te_debug_size(%d): %d", idx, size)
	}
	out := make([]float32, size)
	rc, _, _ := w.procDebugRead.Call(
		uintptr(idx),
		uintptr(unsafe.Pointer(&out[0])),
		uintptr(size),
	)
	if err := w.check("te_debug_read", rc); err != nil {
		return nil, err
	}
	return out, nil
}

func (w *winImpl) init(device int) error {
	rc, _, _ := w.procInit.Call(uintptr(device))
	return w.check("te_init", rc)
}

func (w *winImpl) deviceInfo() (string, error) {
	buf := make([]byte, 256)
	rc, _, _ := w.procDeviceInfo.Call(
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(len(buf)),
	)
	if err := w.check("te_device_info", rc); err != nil {
		return "", err
	}
	return string(buf[:cstrlen(buf)]), nil
}

func (w *winImpl) smokeVectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("length mismatch: %d vs %d", len(a), len(b))
	}
	if len(a) == 0 {
		return nil, nil
	}
	out := make([]float32, len(a))
	rc, _, _ := w.procSmokeVectorAdd.Call(
		uintptr(unsafe.Pointer(&a[0])),
		uintptr(unsafe.Pointer(&b[0])),
		uintptr(unsafe.Pointer(&out[0])),
		uintptr(len(a)),
	)
	if err := w.check("te_smoke_vector_add", rc); err != nil {
		return nil, err
	}
	return out, nil
}

func (w *winImpl) close() error { return w.dll.Release() }

