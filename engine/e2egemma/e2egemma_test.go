// Gemma 3 architecture vs the HF oracle: (1+w) norms, embedding scaling,
// qk-norm, sandwich norms, GELU-tanh, sliding-window layers with dual rope
// theta. The tiny model + dumps are generated on Modal (transformers >= 4.50
// needed for Gemma3 classes) — this skips locally until those dirs exist.
package e2egemma

import (
	"testing"

	"kllm/engine/oracle"
)

func TestGemma3MatchesHF(t *testing.T) {
	oracle.Run(t,
		oracle.RepoPath("testmodels", "tiny-gemma3"),
		oracle.RepoPath("refdumps", "tiny-gemma3"),
		2e-4, 2e-3)
}
