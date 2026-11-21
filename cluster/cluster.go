// Package cluster is the shared vocabulary of kllm's distributed-inference
// layer: the wire types exchanged between the orchestrator and inference
// nodes, the Decision an aggregator produces, and the sequence utilities
// (voting keys, edit similarity, confidence summaries) the rest of the tree
// builds on.
//
// The layer answers one question: when the same prompt is sent to several
// replicas of a model and some of them are wrong — because they are lossy
// (quantized), faulty, slow, or actively adversarial — how should the
// outputs be combined, and what does that cost?
//
//	cluster/node          one inference node: backend + fault injection + HTTP
//	cluster/orchestrator  fan-out, dispatch policies, membership, health
//	cluster/record        JSONL request log + system-metrics sampler
//	cluster/dataset       workloads, scenarios, preprocessing into datasets
//	cluster/aggregate     voting / ranking / confidence / adaptive strategies
//	cluster/detect        outlier, collusion and drift detection
//	cluster/eval          metrics and the scenario sweep that compares it all
//
// kllm works on token ids end to end, so every type here does too.
package cluster

// Request is one inference job as a node sees it.
type Request struct {
	ID           string  `json:"id"`
	Prompt       []int32 `json:"prompt"`
	MaxNewTokens int     `json:"max_new_tokens"`
}

// Response is one node's answer to one Request.
//
// Conf is self-reported: the probability the node's model put on each token
// it emitted. An honest node reports it truthfully; nothing forces a
// dishonest one to, which is exactly what confidence-based aggregation has
// to survive.
type Response struct {
	RequestID string    `json:"request_id"`
	Node      string    `json:"node"`
	Model     string    `json:"model,omitempty"`
	Tokens    []int32   `json:"tokens"`
	Conf      []float32 `json:"conf,omitempty"`
	LatencyMs float64   `json:"latency_ms"`
	Err       string    `json:"error,omitempty"`

	// Label is lab-only ground truth about what the node did to this
	// response. It exists so datasets can be scored; aggregation and
	// detection code never sees it (see Blind).
	Label *Label `json:"label,omitempty"`
}

// OK reports whether the node produced an answer at all.
func (r Response) OK() bool { return r.Err == "" }

// Label is the ground truth a lab node attaches to a response.
type Label struct {
	Fault    string `json:"fault"`    // fault profile configured on the node; "none" if honest
	Attacked bool   `json:"attacked"` // this particular response was tampered with or dropped
}

// Decision is an aggregator's verdict on one request.
type Decision struct {
	Tokens    []int32  `json:"tokens"`
	Winners   []string `json:"winners,omitempty"`   // nodes whose answer is exactly Tokens
	Share     float64  `json:"share"`               // weight behind the winner, as a fraction of all weight cast
	Margin    float64  `json:"margin"`              // winner share minus runner-up share
	Abstained bool     `json:"abstained,omitempty"` // no answer met the strategy's bar
	Suspects  []string `json:"suspects,omitempty"`  // nodes the strategy treated as outliers on this request
}

// Aggregator combines the responses of several nodes into one answer.
// Implementations live in cluster/aggregate. Stateful strategies (the
// adaptive ones learn node reputations) must be safe for concurrent use and
// see requests in arrival order.
type Aggregator interface {
	Name() string
	Aggregate(req Request, resps []Response) Decision
}

// Blind returns copies of resps with the lab labels removed. Everything that
// decides or detects is handed blinded responses, so no strategy can score
// well by reading the answer key.
func Blind(resps []Response) []Response {
	out := make([]Response, len(resps))
	for i, r := range resps {
		r.Label = nil
		out[i] = r
	}
	return out
}

// Answered returns the responses that carry an answer (no error).
func Answered(resps []Response) []Response {
	out := make([]Response, 0, len(resps))
	for _, r := range resps {
		if r.OK() {
			out = append(out, r)
		}
	}
	return out
}
