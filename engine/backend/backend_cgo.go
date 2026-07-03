//go:build linux

package backend

// The Linux loader links libtoyengine.so directly via cgo (the Windows lab
// box uses syscall.LoadDLL instead — see backend_windows.go). Same C-ABI.

/*
#cgo CFLAGS: -I${SRCDIR}/../../backend
#cgo LDFLAGS: -L${SRCDIR}/../../build -ltoyengine -Wl,-rpath,${SRCDIR}/../../build
#include <stdint.h>
#include <stdlib.h>
#include "shim.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

type cgoImpl struct{}

// load ignores path: the library is linked at build time.
func load(string) (impl, error) { return cgoImpl{}, nil }

func lastError() string {
	buf := make([]byte, 512)
	C.te_last_error((*C.char)(unsafe.Pointer(&buf[0])), 512)
	return string(buf[:cstrlen(buf)])
}

func check(fn string, rc C.int) error {
	if rc == 0 {
		return nil
	}
	return fmt.Errorf("%s: backend error %d: %s", fn, int(rc), lastError())
}

func (cgoImpl) init(device int) error {
	return check("te_init", C.te_init(C.int(device)))
}

func (cgoImpl) deviceInfo() (string, error) {
	buf := make([]byte, 256)
	if err := check("te_device_info",
		C.te_device_info((*C.char)(unsafe.Pointer(&buf[0])), 256)); err != nil {
		return "", err
	}
	return string(buf[:cstrlen(buf)]), nil
}

func (cgoImpl) smokeVectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("length mismatch: %d vs %d", len(a), len(b))
	}
	if len(a) == 0 {
		return nil, nil
	}
	out := make([]float32, len(a))
	err := check("te_smoke_vector_add", C.te_smoke_vector_add(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int(len(a))))
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (cgoImpl) modelCreate(cfg *ModelConfig) error {
	return check("te_model_create",
		C.te_model_create((*C.TeModelConfig)(unsafe.Pointer(cfg))))
}

func (cgoImpl) loadTensor(name string, f32raw []byte) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return check("te_model_load_tensor", C.te_model_load_tensor(
		cname,
		(*C.float)(unsafe.Pointer(&f32raw[0])),
		C.int64_t(len(f32raw)/4)))
}

func (cgoImpl) loadTensorW4(name string, q, scales []byte, outDim, inDim, group int64) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return check("te_model_load_tensor_w4", C.te_model_load_tensor_w4(
		cname,
		(*C.uint8_t)(unsafe.Pointer(&q[0])),
		(*C.float)(unsafe.Pointer(&scales[0])),
		C.int64_t(outDim), C.int64_t(inDim), C.int64_t(group)))
}

func (cgoImpl) benchMatmul(m, k, n, iters, mode int64) (float64, error) {
	var ms C.double
	err := check("te_bench_matmul", C.te_bench_matmul(
		C.int64_t(m), C.int64_t(k), C.int64_t(n), C.int64_t(iters),
		C.int64_t(mode), &ms))
	return float64(ms), err
}

func (cgoImpl) setLayerSliding(flags []int32) error {
	return check("te_model_set_layer_sliding", C.te_model_set_layer_sliding(
		(*C.int32_t)(unsafe.Pointer(&flags[0])),
		C.int64_t(len(flags))))
}

func (cgoImpl) finalize() error {
	return check("te_model_finalize", C.te_model_finalize())
}

func (cgoImpl) forwardBatch(seqs []SeqForward, logits []float32) error {
	tokens, nTokens, pos, tables, mbps := flattenBatch(seqs)
	return check("te_forward_batch", C.te_forward_batch(
		C.int64_t(len(seqs)),
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		(*C.int32_t)(unsafe.Pointer(&nTokens[0])),
		(*C.int32_t)(unsafe.Pointer(&pos[0])),
		(*C.int32_t)(unsafe.Pointer(&tables[0])),
		C.int64_t(mbps),
		(*C.float)(unsafe.Pointer(&logits[0]))))
}

func (cgoImpl) setFusion(on bool) error {
	v := C.int64_t(0)
	if on {
		v = 1
	}
	return check("te_set_fusion", C.te_set_fusion(v))
}

func (cgoImpl) debugSet(on bool) error {
	v := C.int64_t(0)
	if on {
		v = 1
	}
	return check("te_debug_set", C.te_debug_set(v))
}

func (cgoImpl) debugCount() (int, error) {
	n := int64(C.te_debug_count())
	if n < 0 {
		return 0, fmt.Errorf("te_debug_count: %d (no model?)", n)
	}
	return int(n), nil
}

func (cgoImpl) debugRead(idx int) ([]float32, error) {
	size := int64(C.te_debug_size(C.int64_t(idx)))
	if size < 0 {
		return nil, fmt.Errorf("te_debug_size(%d): %d", idx, size)
	}
	out := make([]float32, size)
	err := check("te_debug_read", C.te_debug_read(
		C.int64_t(idx),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.int64_t(size)))
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (cgoImpl) close() error { return nil }
