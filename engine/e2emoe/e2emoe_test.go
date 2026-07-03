// Mixtral-style MoE (softmax top-k renorm routing) vs the HF oracle.
// Own package = own process (one model per process).
package e2emoe

import (
	"testing"

	"kllm/engine/oracle"
)

func TestMoESoftmaxMatchesHF(t *testing.T) {
	oracle.Run(t,
		oracle.RepoPath("testmodels", "tiny-mixtral"),
		oracle.RepoPath("refdumps", "tiny-mixtral"),
		2e-4, 2e-3)
}
