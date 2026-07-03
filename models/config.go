// Package models parses HF config.json files into backend model configs.
package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"kllm/engine/backend"
)

// HFConfig is the subset of HF config.json the engine needs.
type HFConfig struct {
	Architectures         []string `json:"architectures"`
	ModelType             string   `json:"model_type"`
	HiddenSize            int64    `json:"hidden_size"`
	IntermediateSize      int64    `json:"intermediate_size"`
	NumHiddenLayers       int64    `json:"num_hidden_layers"`
	NumAttentionHeads     int64    `json:"num_attention_heads"`
	NumKeyValueHeads      int64    `json:"num_key_value_heads"`
	HeadDim               int64    `json:"head_dim"`
	VocabSize             int64    `json:"vocab_size"`
	MaxPositionEmbeddings int64    `json:"max_position_embeddings"`
	RopeTheta             float64  `json:"rope_theta"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	EOSTokenID            int64    `json:"eos_token_id"`

	// MoE (Mixtral naming; 0 = dense)
	NumLocalExperts  int64 `json:"num_local_experts"`
	NumExpertsPerTok int64 `json:"num_experts_per_tok"`
	// kllm extension: "sigmoid_bias" selects sigmoid + expert-bias routing
	// (Sarvam/DeepSeek-V3 family) instead of softmax top-k.
	KllmRouter string `json:"kllm_router"`

	// Gemma 3 (model_type gemma3_text)
	HiddenActivation     string          `json:"hidden_activation"`
	SlidingWindow        int64           `json:"sliding_window"`
	SlidingWindowPattern int64           `json:"sliding_window_pattern"`
	LayerTypes           []string        `json:"layer_types"`
	RopeLocalBaseFreq    float64         `json:"rope_local_base_freq"`
	QueryPreAttnScalar   float64         `json:"query_pre_attn_scalar"`
	RopeScaling          json.RawMessage `json:"rope_scaling"`
	// transformers >= 5 stops writing top-level rope_theta /
	// rope_local_base_freq and nests them here instead (per layer type for
	// hybrid-attention models). Parsed in LoadConfig.
	RopeParameters json.RawMessage `json:"rope_parameters"`
}

type ropeParams struct {
	RopeTheta float64 `json:"rope_theta"`
	RopeType  string  `json:"rope_type"`
}

// applyRopeParameters folds the transformers-5 rope_parameters block into
// the flat RopeTheta / RopeLocalBaseFreq fields the engine uses.
func (c *HFConfig) applyRopeParameters() error {
	if len(c.RopeParameters) == 0 || string(c.RopeParameters) == "null" {
		return nil
	}
	check := func(p ropeParams, key string) error {
		if p.RopeType != "" && p.RopeType != "default" {
			return fmt.Errorf("rope_parameters.%s rope_type %q not supported (no rope scaling yet)", key, p.RopeType)
		}
		return nil
	}
	// Hybrid form: {"full_attention": {...}, "sliding_attention": {...}}
	var keyed map[string]ropeParams
	if err := json.Unmarshal(c.RopeParameters, &keyed); err == nil {
		full, hasFull := keyed["full_attention"]
		slide, hasSlide := keyed["sliding_attention"]
		if hasFull || hasSlide {
			if hasFull && full.RopeTheta > 0 {
				if err := check(full, "full_attention"); err != nil {
					return err
				}
				c.RopeTheta = full.RopeTheta
			}
			if hasSlide && slide.RopeTheta > 0 {
				if err := check(slide, "sliding_attention"); err != nil {
					return err
				}
				c.RopeLocalBaseFreq = slide.RopeTheta
			}
			return nil
		}
	}
	// Flat form: {"rope_theta": ..., "rope_type": ...}
	var flat ropeParams
	if err := json.Unmarshal(c.RopeParameters, &flat); err == nil && flat.RopeTheta > 0 {
		if err := check(flat, ""); err != nil {
			return err
		}
		c.RopeTheta = flat.RopeTheta
		return nil
	}
	return fmt.Errorf("unrecognized rope_parameters shape: %s", c.RopeParameters)
}

// LoadConfig reads <dir>/config.json and applies HF defaults.
func LoadConfig(dir string) (*HFConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	c := &HFConfig{RopeTheta: 10000.0, RMSNormEps: 1e-6, EOSTokenID: -1}
	if err := json.Unmarshal(raw, c); err != nil {
		return nil, fmt.Errorf("parsing config.json: %w", err)
	}
	if c.HiddenSize <= 0 || c.NumHiddenLayers <= 0 || c.NumAttentionHeads <= 0 || c.VocabSize <= 0 {
		return nil, fmt.Errorf("config.json missing required model dims")
	}
	if c.NumKeyValueHeads == 0 {
		c.NumKeyValueHeads = c.NumAttentionHeads
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	if err := c.applyRopeParameters(); err != nil {
		return nil, err
	}
	return c, nil
}

// Backend converts to the C-ABI config. maxSeq bounds KV allocation; 0 uses
// the model's max_position_embeddings.
func (c *HFConfig) Backend(maxSeq int64) backend.ModelConfig {
	if maxSeq == 0 {
		maxSeq = c.MaxPositionEmbeddings
	}
	cfg := backend.ModelConfig{
		Hidden:       c.HiddenSize,
		NLayers:      c.NumHiddenLayers,
		NHeads:       c.NumAttentionHeads,
		NKVHeads:     c.NumKeyValueHeads,
		HeadDim:      c.HeadDim,
		Intermediate: c.IntermediateSize,
		Vocab:        c.VocabSize,
		MaxSeq:       maxSeq,
		NExperts:     c.NumLocalExperts,
		TopK:         c.NumExpertsPerTok,
		RopeTheta:    c.RopeTheta,
		RMSEps:       c.RMSNormEps,
	}
	if cfg.NExperts > 0 {
		cfg.MoeIntermediate = c.IntermediateSize // Mixtral: intermediate_size = expert width
		if c.KllmRouter == "sigmoid_bias" {
			cfg.RouterMode = 1
		}
	}
	if strings.HasPrefix(c.ModelType, "gemma3") {
		cfg.Arch = 1
		cfg.SlidingWindow = c.SlidingWindow
		cfg.SlidingPattern = c.SlidingWindowPattern
		if cfg.SlidingPattern == 0 && len(c.LayerTypes) > 0 {
			// Placeholder to pass create-time validation; the engine sends
			// the explicit per-layer flags right after ModelCreate.
			cfg.SlidingPattern = c.NumHiddenLayers + 1
		}
		cfg.RopeLocalTheta = c.RopeLocalBaseFreq
		cfg.QueryScalar = c.QueryPreAttnScalar
	}
	return cfg
}

// LayerSlidingFlags converts layer_types into per-layer flags for the
// backend (1 = sliding). nil when the checkpoint doesn't ship layer_types
// (the backend then falls back to the (l+1)%pattern formula).
func (c *HFConfig) LayerSlidingFlags() []int32 {
	if len(c.LayerTypes) == 0 {
		return nil
	}
	flags := make([]int32, len(c.LayerTypes))
	for i, lt := range c.LayerTypes {
		if lt == "sliding_attention" {
			flags[i] = 1
		}
	}
	return flags
}

// Validate rejects configs the backend would silently mis-run.
func (c *HFConfig) Validate() error {
	if strings.HasPrefix(c.ModelType, "gemma3") {
		if c.HiddenActivation != "" && c.HiddenActivation != "gelu_pytorch_tanh" {
			return fmt.Errorf("gemma3 activation %q not supported", c.HiddenActivation)
		}
		if len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null" {
			return fmt.Errorf("rope_scaling is set; the backend does not implement it yet")
		}
		if len(c.LayerTypes) > 0 && int64(len(c.LayerTypes)) != c.NumHiddenLayers {
			return fmt.Errorf("layer_types has %d entries for %d layers", len(c.LayerTypes), c.NumHiddenLayers)
		}
		if c.SlidingWindow > 0 && len(c.LayerTypes) == 0 && c.SlidingWindowPattern <= 0 {
			return fmt.Errorf("sliding_window set but neither layer_types nor sliding_window_pattern present")
		}
	}
	return nil
}
