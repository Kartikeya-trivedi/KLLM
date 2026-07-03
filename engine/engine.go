// Package engine ties the pieces together: load a checkpoint through the
// backend, manage paged-KV sequences, and run decode.
package engine

import (
	"fmt"

	"kllm/engine/backend"
	"kllm/engine/kv"
	"kllm/engine/loader"
	"kllm/models"
)

// Options sizes the engine. Zero values get sensible lab defaults.
type Options struct {
	Device     int
	MaxSeq     int64 // per-sequence cap AND max concatenated tokens per step
	BlockSize  int   // KV block size in tokens
	NumBlocks  int   // KV pool size; default fits ~4 max-length sequences
}

func (o *Options) fill() {
	if o.MaxSeq == 0 {
		o.MaxSeq = 256
	}
	if o.BlockSize == 0 {
		o.BlockSize = 16
	}
	if o.NumBlocks == 0 {
		o.NumBlocks = 4 * int(o.MaxSeq) / o.BlockSize
	}
}

type Engine struct {
	B     *backend.Handle
	Cfg   *models.HFConfig
	Alloc *kv.Allocator

	maxStepTokens int // backend scratch bound: total tokens per forward step
}

// MaxStepTokens is the cap on concatenated tokens in one forward step.
func (e *Engine) MaxStepTokens() int { return e.maxStepTokens }

// New loads the backend library, creates the model from modelDir
// (config.json + safetensors), uploads all weights, and finalizes.
func New(dllPath, modelDir string, opts Options) (*Engine, error) {
	opts.fill()
	cfg, err := models.LoadConfig(modelDir)
	if err != nil {
		return nil, err
	}
	h, err := backend.Load(dllPath, opts.Device)
	if err != nil {
		return nil, err
	}
	bcfg := cfg.Backend(opts.MaxSeq)
	bcfg.KVBlockSize = int64(opts.BlockSize)
	bcfg.KVNumBlocks = int64(opts.NumBlocks)
	if err := h.ModelCreate(bcfg); err != nil {
		h.Close()
		return nil, err
	}

	m, err := loader.OpenModel(modelDir)
	if err != nil {
		h.Close()
		return nil, err
	}
	defer m.Close()
	for _, t := range m.Tensors() {
		if t.Dtype != loader.F32 {
			h.Close()
			return nil, fmt.Errorf("tensor %s is %s; only F32 checkpoints supported so far", t.Name, t.Dtype)
		}
		raw, err := m.ReadTensor(t.Name)
		if err != nil {
			h.Close()
			return nil, err
		}
		if err := h.LoadTensorF32(t.Name, raw); err != nil {
			h.Close()
			return nil, fmt.Errorf("uploading %s: %w", t.Name, err)
		}
	}
	if err := h.Finalize(); err != nil {
		h.Close()
		return nil, err
	}
	return &Engine{
		B:             h,
		Cfg:           cfg,
		Alloc:         kv.NewAllocator(opts.NumBlocks, opts.BlockSize),
		maxStepTokens: int(opts.MaxSeq),
	}, nil
}

// Sequence is one generation stream: its KV block table plus position.
type Sequence struct {
	e   *Engine
	kvs *kv.Sequence
}

func (e *Engine) NewSequence() *Sequence {
	return &Sequence{e: e, kvs: e.Alloc.NewSequence()}
}

// Step exposes the sequence's batch descriptor for the given new tokens,
// reserving KV blocks. The scheduler composes these into ForwardBatch calls;
// call Commit after the forward succeeds.
func (s *Sequence) Step(tokens []int32) (backend.SeqForward, error) {
	if err := s.kvs.Reserve(len(tokens)); err != nil {
		return backend.SeqForward{}, err
	}
	return backend.SeqForward{Tokens: tokens, Pos: s.kvs.Len, BlockTable: s.kvs.Table}, nil
}

// Commit records tokens written by a successful forward.
func (s *Sequence) Commit(n int) { s.kvs.Commit(n) }

// Len is the sequence's current KV length in tokens.
func (s *Sequence) Len() int { return s.kvs.Len }

// Forward runs this sequence alone (batch of one) and commits.
func (s *Sequence) Forward(tokens []int32) ([]float32, error) {
	sf, err := s.Step(tokens)
	if err != nil {
		return nil, err
	}
	logits, err := s.e.B.ForwardBatch([]backend.SeqForward{sf})
	if err != nil {
		return nil, err
	}
	s.Commit(len(tokens))
	return logits[0], nil
}

// Release returns the sequence's KV blocks to the pool.
func (s *Sequence) Release() { s.kvs.Release() }

// Generate greedy-decodes up to maxNewTokens after the prompt, stopping at
// the model's EOS id (if it has one).
func (e *Engine) Generate(prompt []int32, maxNewTokens int) ([]int32, error) {
	seq := e.NewSequence()
	defer seq.Release()

	logits, err := seq.Forward(prompt)
	if err != nil {
		return nil, err
	}
	var out []int32
	for range maxNewTokens {
		next := Argmax(logits)
		out = append(out, next)
		if e.Cfg.EOSTokenID >= 0 && int64(next) == e.Cfg.EOSTokenID {
			break
		}
		logits, err = seq.Forward([]int32{next})
		if err != nil {
			return out, err
		}
	}
	return out, nil
}

// Argmax returns the index of the largest logit (greedy sampling).
func Argmax(logits []float32) int32 {
	best, bestVal := 0, logits[0]
	for i, v := range logits[1:] {
		if v > bestVal {
			best, bestVal = i+1, v
		}
	}
	return int32(best)
}

func (e *Engine) Close() error { return e.B.Close() }
