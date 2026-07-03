// Package models parses HF config.json files into backend model configs.
package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

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
	return cfg
}
