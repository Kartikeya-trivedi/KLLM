package loader

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// stEntry is a tensor to place in a test safetensors file.
type stEntry struct {
	name  string
	dtype Dtype
	shape []int64
	data  []byte
}

// writeST builds a valid safetensors file from entries, in order.
func writeST(t *testing.T, path string, meta map[string]string, entries []stEntry) {
	t.Helper()
	header := map[string]any{}
	if meta != nil {
		header["__metadata__"] = meta
	}
	var data []byte
	for _, e := range entries {
		begin := len(data)
		data = append(data, e.data...)
		header[e.name] = map[string]any{
			"dtype":        string(e.dtype),
			"shape":        e.shape,
			"data_offsets": []int{begin, len(data)},
		}
	}
	hj, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(hj)))
	buf = append(buf, hj...)
	buf = append(buf, data...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

func f32bytes(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
	}
	return out
}

func TestOpenAndReadTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	a := f32bytes(1, 2, 3, 4, 5, 6)
	b := []byte{0x00, 0x3f, 0x80, 0x3f, 0x00, 0x40, 0x40, 0x40} // 4 x BF16
	writeST(t, path, map[string]string{"format": "pt"}, []stEntry{
		{"model.a", F32, []int64{2, 3}, a},
		{"model.b", BF16, []int64{4}, b},
	})

	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if f.Metadata["format"] != "pt" {
		t.Errorf("metadata = %v, want format:pt", f.Metadata)
	}
	if got := len(f.Tensors()); got != 2 {
		t.Fatalf("got %d tensors, want 2", got)
	}

	ta, ok := f.Tensor("model.a")
	if !ok {
		t.Fatal("model.a not found")
	}
	if ta.NumElements() != 6 || ta.NumBytes() != 24 || ta.Dtype != F32 {
		t.Errorf("model.a: numel=%d bytes=%d dtype=%s", ta.NumElements(), ta.NumBytes(), ta.Dtype)
	}
	got, err := f.ReadTensor("model.a")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(a) {
		t.Error("model.a bytes round-trip mismatch")
	}
	got, err = f.ReadTensor("model.b")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(b) {
		t.Error("model.b bytes round-trip mismatch")
	}

	if _, err := f.ReadTensor("nope"); err == nil {
		t.Error("reading missing tensor should fail")
	}
	if err := f.ReadTensorInto("model.a", make([]byte, 3)); err == nil {
		t.Error("wrong-size dst should fail")
	}
}

func TestOpenRejectsCorruptFiles(t *testing.T) {
	dir := t.TempDir()

	cases := []struct {
		name  string
		write func(path string)
	}{
		{"truncated header length", func(p string) {
			os.WriteFile(p, []byte{1, 2, 3}, 0o644)
		}},
		{"implausible header length", func(p string) {
			buf := make([]byte, 8)
			binary.LittleEndian.PutUint64(buf, 1<<40)
			os.WriteFile(p, append(buf, []byte("{}")...), 0o644)
		}},
		{"offsets past end of file", func(p string) {
			hj := `{"t":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}`
			buf := make([]byte, 8)
			binary.LittleEndian.PutUint64(buf, uint64(len(hj)))
			buf = append(buf, hj...)
			buf = append(buf, make([]byte, 16)...) // only 16 of 32 data bytes
			os.WriteFile(p, buf, 0o644)
		}},
		{"shape and byte-length disagree", func(p string) {
			hj := `{"t":{"dtype":"F32","shape":[3],"data_offsets":[0,16]}}`
			buf := make([]byte, 8)
			binary.LittleEndian.PutUint64(buf, uint64(len(hj)))
			buf = append(buf, hj...)
			buf = append(buf, make([]byte, 16)...)
			os.WriteFile(p, buf, 0o644)
		}},
		{"unknown dtype", func(p string) {
			hj := `{"t":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[0,4]}}`
			buf := make([]byte, 8)
			binary.LittleEndian.PutUint64(buf, uint64(len(hj)))
			buf = append(buf, hj...)
			buf = append(buf, make([]byte, 4)...)
			os.WriteFile(p, buf, 0o644)
		}},
	}
	for i, c := range cases {
		p := filepath.Join(dir, fmt.Sprintf("bad%d.safetensors", i))
		c.write(p)
		if _, err := Open(p); err == nil {
			t.Errorf("%s: Open succeeded, want error", c.name)
		}
	}
}

func TestOpenModelSharded(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), nil, []stEntry{
		{"layer.0.w", F32, []int64{2}, f32bytes(1, 2)},
	})
	writeST(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), nil, []stEntry{
		{"layer.1.w", F32, []int64{2}, f32bytes(3, 4)},
	})
	index := map[string]any{"weight_map": map[string]string{
		"layer.0.w": "model-00001-of-00002.safetensors",
		"layer.1.w": "model-00002-of-00002.safetensors",
	}}
	ij, _ := json.Marshal(index)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ij, 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if len(m.Files) != 2 {
		t.Fatalf("got %d shards, want 2", len(m.Files))
	}
	names := []string{}
	for _, tn := range m.Tensors() {
		names = append(names, tn.Name)
	}
	if strings.Join(names, ",") != "layer.0.w,layer.1.w" {
		t.Errorf("tensor names = %v", names)
	}
	got, err := m.ReadTensor("layer.1.w")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(f32bytes(3, 4)) {
		t.Error("layer.1.w bytes mismatch")
	}
}

func TestOpenModelDuplicateTensor(t *testing.T) {
	dir := t.TempDir()
	writeST(t, filepath.Join(dir, "a.safetensors"), nil, []stEntry{
		{"w", F32, []int64{1}, f32bytes(1)},
	})
	writeST(t, filepath.Join(dir, "b.safetensors"), nil, []stEntry{
		{"w", F32, []int64{1}, f32bytes(2)},
	})
	if _, err := OpenModel(dir); err == nil {
		t.Error("duplicate tensor across shards should fail")
	}
}

func TestOpenModelSingleFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeST(t, path, nil, []stEntry{{"w", F16, []int64{2}, []byte{1, 2, 3, 4}}})
	m, err := OpenModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if _, ok := m.Tensor("w"); !ok {
		t.Error("tensor w not found via single-file OpenModel")
	}
}
