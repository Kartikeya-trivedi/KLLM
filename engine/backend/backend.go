// Package backend is the Go side of the narrow C-ABI boundary to the CUDA
// backend (backend/shim.h). Small control data crosses the boundary; all big
// tensors stay on-device.
//
// Two implementations are planned behind this API:
//   - backend_windows.go: LoadLibrary + GetProcAddress via syscall (no cgo,
//     avoids the MinGW-w64 requirement on the Windows kernel-lab box)
//   - backend_cgo.go (linux): plain cgo against libtoyengine.so on the A6000
//     engine box
package backend

// Handle is an initialized connection to the CUDA backend library.
type Handle struct {
	impl impl
}

// impl is what each platform loader must provide.
type impl interface {
	init(device int) error
	deviceInfo() (string, error)
	smokeVectorAdd(a, b []float32) ([]float32, error)
	close() error
}

// Load opens the backend shared library at path and creates the CUDA context
// on the given device.
func Load(path string, device int) (*Handle, error) {
	im, err := load(path) // platform-specific (backend_windows.go / backend_cgo.go)
	if err != nil {
		return nil, err
	}
	if err := im.init(device); err != nil {
		im.close()
		return nil, err
	}
	return &Handle{impl: im}, nil
}

// DeviceInfo returns a one-line description of the active CUDA device.
func (h *Handle) DeviceInfo() (string, error) { return h.impl.deviceInfo() }

// SmokeVectorAdd runs the walking-skeleton kernel: element-wise a+b on the GPU.
func (h *Handle) SmokeVectorAdd(a, b []float32) ([]float32, error) {
	return h.impl.smokeVectorAdd(a, b)
}

// Close releases the library handle. The CUDA context lives until process exit.
func (h *Handle) Close() error { return h.impl.close() }
