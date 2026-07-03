// Package loader reads safetensors checkpoints (single-file or sharded with
// a model.safetensors.index.json) into host memory. Format: 8-byte LE u64
// header length, JSON header mapping tensor name -> {dtype, shape,
// data_offsets}, then one raw byte buffer.
//
// The loader stays raw: it hands out bytes + dtype + shape. Dtype conversion
// and device upload are the backend's job.
package loader

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Dtype is a safetensors dtype string.
type Dtype string

const (
	F64  Dtype = "F64"
	F32  Dtype = "F32"
	F16  Dtype = "F16"
	BF16 Dtype = "BF16"
	I64  Dtype = "I64"
	I32  Dtype = "I32"
	I16  Dtype = "I16"
	I8   Dtype = "I8"
	U8   Dtype = "U8"
	Bool Dtype = "BOOL"
)

var dtypeSize = map[Dtype]int64{
	F64: 8, F32: 4, F16: 2, BF16: 2,
	I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, Bool: 1,
}

// Size returns the element size in bytes, or an error for unknown dtypes.
func (d Dtype) Size() (int64, error) {
	if s, ok := dtypeSize[d]; ok {
		return s, nil
	}
	return 0, fmt.Errorf("unsupported safetensors dtype %q", d)
}

// Tensor describes one tensor within a safetensors file.
type Tensor struct {
	Name  string
	Dtype Dtype
	Shape []int64

	begin, end int64 // absolute byte offsets within the file
}

// NumElements returns the product of the shape (1 for scalars/empty shape).
func (t *Tensor) NumElements() int64 {
	n := int64(1)
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// NumBytes returns the tensor's size in the file.
func (t *Tensor) NumBytes() int64 { return t.end - t.begin }

// File is one opened .safetensors file.
type File struct {
	Path     string
	Metadata map[string]string

	f       *os.File
	tensors map[string]*Tensor
}

// maxHeaderLen guards against garbage in the 8-byte length prefix.
const maxHeaderLen = 512 << 20

type headerEntry struct {
	Dtype       Dtype    `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Open parses a safetensors file's header and validates every tensor's
// offsets against the file size. Tensor data is read lazily via ReadTensor.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	sf, err := parse(f, path)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return sf, nil
}

func parse(f *os.File, path string) (*File, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := st.Size()

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, fmt.Errorf("reading header length: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen > maxHeaderLen || int64(headerLen) > fileSize-8 {
		return nil, fmt.Errorf("implausible header length %d (file size %d)", headerLen, fileSize)
	}

	headerJSON := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerJSON); err != nil {
		return nil, fmt.Errorf("reading header: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &raw); err != nil {
		return nil, fmt.Errorf("parsing header JSON: %w", err)
	}

	dataStart := 8 + int64(headerLen)
	dataLen := fileSize - dataStart
	sf := &File{Path: path, f: f, tensors: make(map[string]*Tensor, len(raw))}

	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &sf.Metadata); err != nil {
				return nil, fmt.Errorf("parsing __metadata__: %w", err)
			}
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(msg, &e); err != nil {
			return nil, fmt.Errorf("tensor %q: parsing entry: %w", name, err)
		}
		elemSize, err := e.Dtype.Size()
		if err != nil {
			return nil, fmt.Errorf("tensor %q: %w", name, err)
		}
		numel := int64(1)
		for _, d := range e.Shape {
			if d < 0 {
				return nil, fmt.Errorf("tensor %q: negative dim %d", name, d)
			}
			if d != 0 && numel > math.MaxInt64/d {
				return nil, fmt.Errorf("tensor %q: shape overflows int64", name)
			}
			numel *= d
		}
		begin, end := e.DataOffsets[0], e.DataOffsets[1]
		if begin < 0 || begin > end || end > dataLen {
			return nil, fmt.Errorf("tensor %q: data_offsets [%d,%d) out of bounds (data section %d bytes)",
				name, begin, end, dataLen)
		}
		if end-begin != numel*elemSize {
			return nil, fmt.Errorf("tensor %q: %d bytes in file but shape %v of %s needs %d",
				name, end-begin, e.Shape, e.Dtype, numel*elemSize)
		}
		sf.tensors[name] = &Tensor{
			Name:  name,
			Dtype: e.Dtype,
			Shape: e.Shape,
			begin: dataStart + begin,
			end:   dataStart + end,
		}
	}
	return sf, nil
}

// Tensor looks up a tensor by name.
func (f *File) Tensor(name string) (*Tensor, bool) {
	t, ok := f.tensors[name]
	return t, ok
}

// Tensors returns all tensors sorted by name.
func (f *File) Tensors() []*Tensor {
	out := make([]*Tensor, 0, len(f.tensors))
	for _, t := range f.tensors {
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ReadTensor reads a tensor's raw bytes into a fresh buffer.
func (f *File) ReadTensor(name string) ([]byte, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("%s: no tensor %q", f.Path, name)
	}
	buf := make([]byte, t.NumBytes())
	if err := f.ReadTensorInto(name, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

// ReadTensorInto reads a tensor's raw bytes into dst, which must be exactly
// NumBytes() long. Safe for concurrent use (ReadAt).
func (f *File) ReadTensorInto(name string, dst []byte) error {
	t, ok := f.tensors[name]
	if !ok {
		return fmt.Errorf("%s: no tensor %q", f.Path, name)
	}
	if int64(len(dst)) != t.NumBytes() {
		return fmt.Errorf("tensor %q: dst is %d bytes, need %d", name, len(dst), t.NumBytes())
	}
	if _, err := f.f.ReadAt(dst, t.begin); err != nil {
		return fmt.Errorf("tensor %q: %w", name, err)
	}
	return nil
}

func (f *File) Close() error { return f.f.Close() }

// Model is a checkpoint: one or more safetensors files with a unified
// tensor namespace, as produced by HF (sharded checkpoints ship a
// model.safetensors.index.json).
type Model struct {
	Files  []*File
	byName map[string]*File
}

type indexFile struct {
	WeightMap map[string]string `json:"weight_map"`
}

// OpenModel opens a checkpoint from a directory (using the index file if
// present, else every *.safetensors in it) or from a single .safetensors path.
func OpenModel(path string) (*Model, error) {
	st, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	var paths []string
	if !st.IsDir() {
		paths = []string{path}
	} else if idx, err := os.ReadFile(filepath.Join(path, "model.safetensors.index.json")); err == nil {
		var index indexFile
		if err := json.Unmarshal(idx, &index); err != nil {
			return nil, fmt.Errorf("parsing index: %w", err)
		}
		seen := map[string]bool{}
		for _, file := range index.WeightMap {
			if !seen[file] {
				seen[file] = true
				paths = append(paths, filepath.Join(path, file))
			}
		}
		sort.Strings(paths)
	} else {
		entries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".safetensors") {
				paths = append(paths, filepath.Join(path, e.Name()))
			}
		}
		sort.Strings(paths)
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("%s: no .safetensors files found", path)
	}

	m := &Model{byName: make(map[string]*File)}
	for _, p := range paths {
		f, err := Open(p)
		if err != nil {
			m.Close()
			return nil, err
		}
		m.Files = append(m.Files, f)
		for name := range f.tensors {
			if prev, dup := m.byName[name]; dup {
				m.Close()
				return nil, fmt.Errorf("tensor %q appears in both %s and %s", name, prev.Path, p)
			}
			m.byName[name] = f
		}
	}
	return m, nil
}

// Tensor looks up a tensor by name across all shards.
func (m *Model) Tensor(name string) (*Tensor, bool) {
	f, ok := m.byName[name]
	if !ok {
		return nil, false
	}
	return f.Tensor(name)
}

// Tensors returns all tensors across all shards, sorted by name.
func (m *Model) Tensors() []*Tensor {
	out := make([]*Tensor, 0, len(m.byName))
	for _, f := range m.Files {
		out = append(out, f.Tensors()...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ReadTensor reads a tensor's raw bytes from whichever shard holds it.
func (m *Model) ReadTensor(name string) ([]byte, error) {
	f, ok := m.byName[name]
	if !ok {
		return nil, fmt.Errorf("no tensor %q in model", name)
	}
	return f.ReadTensor(name)
}

// Close closes all shard files.
func (m *Model) Close() error {
	var first error
	for _, f := range m.Files {
		if err := f.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
