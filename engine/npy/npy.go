// Package npy reads the minimal subset of the .npy format the correctness
// harness needs: v1.0/v2.0 headers, little-endian float32, C order.
package npy

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

var (
	descrRe = regexp.MustCompile(`'descr':\s*'([^']+)'`)
	orderRe = regexp.MustCompile(`'fortran_order':\s*(True|False)`)
	shapeRe = regexp.MustCompile(`'shape':\s*\(([^)]*)\)`)
)

// ReadF32 loads a float32 .npy file, returning its shape and flat data.
func ReadF32(path string) ([]int, []float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	if len(raw) < 10 || string(raw[:6]) != "\x93NUMPY" {
		return nil, nil, fmt.Errorf("%s: not an npy file", path)
	}
	major := raw[6]
	var headerLen, dataStart int
	switch major {
	case 1:
		headerLen = int(binary.LittleEndian.Uint16(raw[8:10]))
		dataStart = 10 + headerLen
	case 2, 3:
		headerLen = int(binary.LittleEndian.Uint32(raw[8:12]))
		dataStart = 12 + headerLen
	default:
		return nil, nil, fmt.Errorf("%s: unsupported npy version %d", path, major)
	}
	if dataStart > len(raw) {
		return nil, nil, fmt.Errorf("%s: truncated header", path)
	}
	header := string(raw[dataStart-headerLen : dataStart])

	m := descrRe.FindStringSubmatch(header)
	if m == nil || m[1] != "<f4" {
		return nil, nil, fmt.Errorf("%s: unsupported descr (want <f4): %s", path, header)
	}
	if o := orderRe.FindStringSubmatch(header); o == nil || o[1] != "False" {
		return nil, nil, fmt.Errorf("%s: fortran_order not supported", path)
	}
	s := shapeRe.FindStringSubmatch(header)
	if s == nil {
		return nil, nil, fmt.Errorf("%s: no shape in header", path)
	}
	var shape []int
	numel := 1
	for _, part := range strings.Split(s[1], ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		d, err := strconv.Atoi(part)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: bad shape dim %q", path, part)
		}
		shape = append(shape, d)
		numel *= d
	}

	body := raw[dataStart:]
	if len(body) != numel*4 {
		return nil, nil, fmt.Errorf("%s: %d data bytes for %d elements", path, len(body), numel)
	}
	out := make([]float32, numel)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[i*4:]))
	}
	return shape, out, nil
}
