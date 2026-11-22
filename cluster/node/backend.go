// Package node is one inference node of the cluster: a Backend that turns a
// prompt into tokens, an optional fault profile wrapped around it, and the
// HTTP surface the orchestrator talks to.
//
// Three backends ship:
//
//   - Sim, a deterministic synthetic language model. It needs no GPU, has a
//     known ground truth, and its replicas disagree the way real replicas do
//     (at low-margin decode steps), so robustness experiments are cheap and
//     exactly reproducible.
//   - Remote, which forwards to anything speaking the node protocol — in
//     particular a `kllm serve` process, so the real engine can sit behind a
//     fault profile.
//   - kllm itself: `kllm serve` answers POST /v1/infer natively, so an
//     unwrapped engine is already an (honest) cluster node.
package node

import (
	"context"
	"time"

	"kllm/cluster"
)

// FaultNone labels an honest node's output.
const FaultNone = "none"

// Output is what a Backend produces for one request.
type Output struct {
	Tokens []int32
	Conf   []float32 // probability the model put on each emitted token

	// Latency is compute time the backend modelled but did not spend (the
	// simulator never sleeps). A backend that really did the work leaves
	// it zero. The node either sleeps it (realtime) or just reports it
	// (virtual time, for fast dataset generation).
	Latency time.Duration

	// Label is the ground truth about what happened to this output.
	Label cluster.Label
}

// Backend generates a continuation for a request.
type Backend interface {
	// Model names what is answering, for logs and datasets.
	Model() string
	Generate(ctx context.Context, req cluster.Request) (Output, error)
}

// Instant is implemented by a backend whose Generate takes no real time:
// the synthetic model, which reports a latency instead of spending it. A
// Fast adversary can consult such a model and still answer first; in front
// of anything else it has to answer without it.
type Instant interface {
	Instant() bool
}
