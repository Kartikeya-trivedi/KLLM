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

import "fmt"

// ModelConfig crosses the boundary as a raw pointer. Field order and types
// MUST match TeModelConfig in backend/shim.h exactly (all 8-byte fields).
type ModelConfig struct {
	Hidden       int64
	NLayers      int64
	NHeads       int64
	NKVHeads     int64
	HeadDim      int64
	Intermediate int64
	Vocab        int64
	MaxSeq       int64
	// MoE (all 0 for dense)
	NExperts        int64
	TopK            int64
	MoeIntermediate int64
	RouterMode      int64 // 0 = softmax top-k renorm, 1 = sigmoid + expert bias
	// Paged KV
	KVBlockSize int64
	KVNumBlocks int64
	RopeTheta   float64
	RMSEps      float64
}

// MaxBatchSeqs mirrors TE_MAX_BATCH_SEQS in shim.h.
const MaxBatchSeqs = 128

// SeqForward is one sequence's slice of a forward_step batch.
type SeqForward struct {
	Tokens     []int32 // new tokens (prefill: many, decode: one)
	Pos        int     // absolute position of Tokens[0]
	BlockTable []int32 // physical KV block ids covering Pos+len(Tokens)
}

// Handle is an initialized connection to the CUDA backend library.
type Handle struct {
	impl impl
	cfg  ModelConfig // set by ModelCreate; Vocab sizes the logits buffer
}

// impl is what each platform loader must provide.
type impl interface {
	init(device int) error
	deviceInfo() (string, error)
	smokeVectorAdd(a, b []float32) ([]float32, error)
	modelCreate(cfg *ModelConfig) error
	loadTensor(name string, f32raw []byte) error
	finalize() error
	forwardBatch(seqs []SeqForward, logits []float32) error
	setFusion(on bool) error
	debugSet(on bool) error
	debugCount() (int, error)
	debugRead(idx int) ([]float32, error)
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

// ModelCreate allocates the (single) model with the given config.
func (h *Handle) ModelCreate(cfg ModelConfig) error {
	if err := h.impl.modelCreate(&cfg); err != nil {
		return err
	}
	h.cfg = cfg
	return nil
}

// LoadTensorF32 uploads one fp32 weight tensor (raw little-endian bytes) by
// its HF name. len(f32raw) must be a multiple of 4.
func (h *Handle) LoadTensorF32(name string, f32raw []byte) error {
	if len(f32raw) == 0 || len(f32raw)%4 != 0 {
		return fmt.Errorf("tensor %s: byte length %d is not a positive multiple of 4", name, len(f32raw))
	}
	return h.impl.loadTensor(name, f32raw)
}

// Finalize validates weight completeness and allocates KV + scratch.
func (h *Handle) Finalize() error { return h.impl.finalize() }

// ForwardBatch runs one forward step for a batch of sequences and returns
// each sequence's last-token logits.
func (h *Handle) ForwardBatch(seqs []SeqForward) ([][]float32, error) {
	if len(seqs) == 0 || len(seqs) > MaxBatchSeqs {
		return nil, fmt.Errorf("forward: batch of %d sequences (max %d)", len(seqs), MaxBatchSeqs)
	}
	for i, s := range seqs {
		if len(s.Tokens) == 0 {
			return nil, fmt.Errorf("forward: sequence %d has no tokens", i)
		}
	}
	vocab := int(h.cfg.Vocab)
	flat := make([]float32, len(seqs)*vocab)
	if err := h.impl.forwardBatch(seqs, flat); err != nil {
		return nil, err
	}
	out := make([][]float32, len(seqs))
	for i := range out {
		out[i] = flat[i*vocab : (i+1)*vocab : (i+1)*vocab]
	}
	return out, nil
}

// SetFusion toggles fused kernels (default on); the unfused path exists for
// honest before/after benchmarking.
func (h *Handle) SetFusion(on bool) error { return h.impl.setFusion(on) }

// DebugSet toggles per-layer activation capture on te_forward.
func (h *Handle) DebugSet(on bool) error { return h.impl.debugSet(on) }

// DebugCount returns how many activation entries the last forward captured.
func (h *Handle) DebugCount() (int, error) { return h.impl.debugCount() }

// DebugRead returns captured activation entry idx.
func (h *Handle) DebugRead(idx int) ([]float32, error) { return h.impl.debugRead(idx) }

// Close releases the library handle. The CUDA context lives until process exit.
func (h *Handle) Close() error { return h.impl.close() }
