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

// check converts a shim return code into a Go error using te_last_error.
func (w *winImpl) check(fn string, rc uintptr) error {
	if rc == 0 {
		return nil
	}
	buf := make([]byte, 512)
	w.procLastError.Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return fmt.Errorf("%s: cuda error %d: %s", fn, rc, string(buf[:cstrlen(buf)]))
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

func cstrlen(b []byte) int {
	for i, c := range b {
		if c == 0 {
			return i
		}
	}
	return len(b)
}
