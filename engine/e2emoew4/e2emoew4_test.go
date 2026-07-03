// INT4 MoE: expert FFNs quantized (router gate fp32), softmax routing,
// validated against HF running on the dequantized twin checkpoint — the
// W4 grouped-expert path computes exactly dequant*matmul.
package e2emoew4

import (
	"testing"

	"kllm/engine/oracle"
)

func TestMoEW4MatchesDequantReference(t *testing.T) {
	oracle.Run(t,
		oracle.RepoPath("testmodels", "tiny-mixtral-w4"),
		oracle.RepoPath("refdumps", "tiny-mixtral-w4dq"),
		2e-4, 2e-3)
}
