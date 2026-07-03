// Package kv is the Go side of paged KV: a block allocator with a free list
// and per-sequence block tables. The CUDA backend owns the block *pool*;
// this package decides which physical blocks each sequence uses.
package kv

import "fmt"

// Allocator hands out physical block ids from a fixed pool.
type Allocator struct {
	blockSize int
	numBlocks int
	free      []int32 // LIFO free list
}

func NewAllocator(numBlocks, blockSize int) *Allocator {
	a := &Allocator{blockSize: blockSize, numBlocks: numBlocks}
	a.free = make([]int32, numBlocks)
	for i := range a.free {
		// LIFO: hand out low ids first for readable debugging.
		a.free[i] = int32(numBlocks - 1 - i)
	}
	return a
}

func (a *Allocator) BlockSize() int  { return a.blockSize }
func (a *Allocator) NumBlocks() int  { return a.numBlocks }
func (a *Allocator) FreeBlocks() int { return len(a.free) }

func (a *Allocator) alloc() (int32, error) {
	if len(a.free) == 0 {
		return 0, fmt.Errorf("kv: out of blocks (%d total)", a.numBlocks)
	}
	id := a.free[len(a.free)-1]
	a.free = a.free[:len(a.free)-1]
	return id, nil
}

func (a *Allocator) release(ids []int32) {
	a.free = append(a.free, ids...)
}

// Sequence tracks one sequence's block table and logical length.
type Sequence struct {
	a     *Allocator
	Table []int32 // physical block ids, in logical order
	Len   int     // tokens written so far
}

func (a *Allocator) NewSequence() *Sequence { return &Sequence{a: a} }

// Reserve ensures the table covers Len+n tokens, allocating blocks as
// needed. On out-of-blocks it leaves already-held blocks in place (the
// caller can Release or retry after others free).
func (s *Sequence) Reserve(n int) error {
	needBlocks := (s.Len + n + s.a.blockSize - 1) / s.a.blockSize
	for len(s.Table) < needBlocks {
		id, err := s.a.alloc()
		if err != nil {
			return err
		}
		s.Table = append(s.Table, id)
	}
	return nil
}

// Commit records that n tokens were written (after a successful forward).
func (s *Sequence) Commit(n int) { s.Len += n }

// Release returns all blocks to the pool and resets the sequence.
func (s *Sequence) Release() {
	s.a.release(s.Table)
	s.Table = nil
	s.Len = 0
}
