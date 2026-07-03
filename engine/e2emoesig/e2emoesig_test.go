// Sigmoid + expert-bias MoE routing (Sarvam/DeepSeek-V3 family) vs the
// numpy oracle. Own package = own process (one model per process).
package e2emoesig

import (
	"testing"

	"kllm/engine/oracle"
)

func TestMoESigmoidMatchesNumpyOracle(t *testing.T) {
	oracle.Run(t,
		oracle.RepoPath("testmodels", "tiny-mixtral-sigmoid"),
		oracle.RepoPath("refdumps", "tiny-mixtral-sigmoid"),
		2e-4, 2e-3)
}
