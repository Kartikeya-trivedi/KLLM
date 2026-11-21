package cluster

import (
	"fmt"
	"sort"
)

// Dispatch modes: how many nodes a request is sent to and how long the
// orchestrator waits. This is where reliability is traded for cost.
const (
	// ModeAll asks every node and waits for all of them (or the timeout).
	// Most evidence, slowest: latency is the slowest node's.
	ModeAll = "all"

	// ModeFirstK asks every node and proceeds once K have answered,
	// cancelling the rest. It cuts tail latency — and hands the decision
	// to whoever answers fastest, which an adversary can arrange to be.
	ModeFirstK = "first-k"

	// ModeAdaptive asks Initial nodes and escalates by Step only while
	// they disagree. Cheap when replicas agree, which is most of the
	// time; the open question is who gets asked first.
	ModeAdaptive = "adaptive"
)

// ErrCancelled marks the response of a node the orchestrator stopped
// waiting for. It is not evidence against the node.
const ErrCancelled = "cancelled: enough nodes answered"

// Policy is a dispatch policy. The orchestrator executes it live; Replay
// applies it after the fact to a recorded run. Both follow these semantics.
type Policy struct {
	Mode      string  `json:"mode"`
	TimeoutMs float64 `json:"timeout_ms"`

	K int `json:"k,omitempty"` // first-k: answers to wait for

	Initial int     `json:"initial,omitempty"` // adaptive: nodes asked first
	Step    int     `json:"step,omitempty"`    // adaptive: nodes added per escalation
	Margin  float64 `json:"margin,omitempty"`  // adaptive: stop when Agreement margin reaches this

	// Order picks who is asked first when not everyone is: "hash" spreads
	// requests evenly; "trust" prefers nodes the aggregator trusts most.
	Order string `json:"order,omitempty"`
}

// Normalize fills defaults and clamps to a cluster of n nodes.
func (p Policy) Normalize(n int) Policy {
	if p.Mode == "" {
		p.Mode = ModeAll
	}
	if p.TimeoutMs <= 0 {
		p.TimeoutMs = 2000
	}
	if p.K <= 0 {
		p.K = n/2 + 1
	}
	p.K = min(p.K, n)
	if p.Initial <= 0 {
		p.Initial = 3
	}
	p.Initial = min(p.Initial, n)
	if p.Step <= 0 {
		p.Step = 2
	}
	if p.Margin <= 0 {
		p.Margin = 1 // unanimous
	}
	if p.Order == "" {
		p.Order = "hash"
	}
	return p
}

func (p Policy) String() string {
	switch p.Mode {
	case ModeFirstK:
		return fmt.Sprintf("first-%d", p.K)
	case ModeAdaptive:
		s := fmt.Sprintf("adaptive-%d+%d", p.Initial, p.Step)
		if p.Order == "trust" {
			s += "-trust"
		}
		return s
	}
	return ModeAll
}

func hash64(s string) uint64 {
	h := uint64(0xcbf29ce484222325)
	for i := 0; i < len(s); i++ {
		h = (h ^ uint64(s[i])) * 0x100000001b3
	}
	h ^= h >> 33
	h *= 0xff51afd7ed558ccd
	return h ^ h>>33
}

// Order returns ids in the order a policy would ask them. "hash" is a
// per-request pseudo-random permutation, so no node is systematically first.
// "trust" sorts by trust, highest first, breaking ties by the hash order.
func Order(p Policy, req Request, ids []string, trust func(string) float64) []string {
	out := append([]string(nil), ids...)
	seed := hash64(req.ID)
	rank := make(map[string]uint64, len(out))
	for _, id := range out {
		rank[id] = hash64(id) ^ seed*0x9e3779b97f4a7c15
	}
	sort.Slice(out, func(i, j int) bool {
		if p.Order == "trust" && trust != nil {
			if ti, tj := trust(out[i]), trust(out[j]); ti != tj {
				return ti > tj
			}
		}
		if rank[out[i]] != rank[out[j]] {
			return rank[out[i]] < rank[out[j]]
		}
		return out[i] < out[j]
	})
	return out
}

// Agreement measures how much a set of responses agree, with no weighting:
// share is the fraction of answers equal to the most common one, margin is
// that minus the runner-up's fraction. answered counts usable responses.
func Agreement(resps []Response) (share, margin float64, answered int) {
	counts := map[string]int{}
	for _, r := range resps {
		if r.OK() {
			counts[Key(r.Tokens)]++
			answered++
		}
	}
	if answered == 0 {
		return 0, 0, 0
	}
	var top, second int
	for _, c := range counts {
		switch {
		case c > top:
			top, second = c, top
		case c > second:
			second = c
		}
	}
	return float64(top) / float64(answered), float64(top-second) / float64(answered), answered
}

// Outcome is what a policy made of one request.
type Outcome struct {
	Used      []Response // what the policy would hand the aggregator
	LatencyMs float64    // time until the policy had its answers
	Queried   int        // nodes asked
	Rounds    int        // adaptive escalation rounds (1 otherwise)
	Tokens    float64    // tokens generated on the policy's behalf — the compute bill
}

// Replay applies the policy to a complete recording of one request — one
// response per node, as a ModeAll run logs them — and reports what the
// policy would have seen and cost. It is how one dataset is scored under
// every dispatch policy without re-running the cluster.
//
// A response slower than the timeout is a timeout. A cancelled first-k
// straggler is billed for the fraction of its answer it had produced.
func (p Policy) Replay(req Request, all []Response, trust func(string) float64) Outcome {
	p = p.Normalize(len(all))
	byNode := make(map[string]Response, len(all))
	ids := make([]string, 0, len(all))
	for _, r := range all {
		if r.OK() && r.LatencyMs > p.TimeoutMs {
			r.Err, r.Tokens, r.Conf = "deadline exceeded", nil, nil
		}
		if r.LatencyMs > p.TimeoutMs {
			r.LatencyMs = p.TimeoutMs
		}
		byNode[r.Node] = r
		ids = append(ids, r.Node)
	}
	var out Outcome
	slowest := func(rs []Response) (ms float64) {
		for _, r := range rs {
			ms = max(ms, r.LatencyMs)
		}
		return
	}
	bill := func(rs []Response) (tokens float64) {
		for _, r := range rs {
			tokens += float64(len(r.Tokens))
		}
		return
	}

	switch p.Mode {
	case ModeFirstK:
		byLatency := make([]Response, 0, len(all))
		for _, id := range ids {
			byLatency = append(byLatency, byNode[id])
		}
		sort.SliceStable(byLatency, func(i, j int) bool { return byLatency[i].LatencyMs < byLatency[j].LatencyMs })
		cut, ok := len(byLatency), 0
		for i, r := range byLatency {
			if r.OK() {
				if ok++; ok == p.K {
					cut = i + 1
					break
				}
			}
		}
		out.Used = byLatency[:cut]
		out.LatencyMs = slowest(out.Used)
		out.Queried, out.Rounds = len(all), 1
		out.Tokens = bill(out.Used)
		for _, r := range byLatency[cut:] { // cancelled mid-generation
			if r.LatencyMs > 0 {
				out.Tokens += float64(len(r.Tokens)) * min(1, out.LatencyMs/r.LatencyMs)
			}
		}

	case ModeAdaptive:
		order := Order(p, req, ids, trust)
		asked := 0
		for asked < len(order) {
			n := p.Step
			if asked == 0 {
				n = p.Initial
			}
			n = min(n, len(order)-asked)
			round := make([]Response, 0, n)
			for _, id := range order[asked : asked+n] {
				round = append(round, byNode[id])
			}
			asked += n
			out.Used = append(out.Used, round...)
			out.LatencyMs += slowest(round)
			out.Rounds++
			if _, margin, answered := Agreement(out.Used); answered >= p.Initial && margin >= p.Margin {
				break
			}
		}
		out.Queried = asked
		out.Tokens = bill(out.Used)

	default: // ModeAll
		for _, id := range ids {
			out.Used = append(out.Used, byNode[id])
		}
		out.LatencyMs = slowest(out.Used)
		out.Queried, out.Rounds = len(all), 1
		out.Tokens = bill(out.Used)
	}
	return out
}
