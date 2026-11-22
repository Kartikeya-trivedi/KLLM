package detect

import (
	"sort"
	"sync"

	"kllm/cluster"
)

// Reasons a node is a suspect.
const (
	Untrusted = "untrusted" // keeps losing the vote
	Drift     = "drift"     // started disagreeing far more than it used to
	Collusion = "collusion" // repeatedly deviates in lockstep with other nodes
)

// Suspect is a node-level verdict.
type Suspect struct {
	Node       string   `json:"node"`
	Reasons    []string `json:"reasons"`
	Trust      float64  `json:"trust"`
	DetectedAt int      `json:"detected_at"` // requests observed when it first became a suspect
}

type nodeState struct {
	seen           int
	agree, dissent float64 // decayed counts
	base           float64 // slow estimate of the node's own dissent rate
	cusum          float64
	drifted        bool
	suspectAt      int // 0 = never
}

type pairState struct {
	joint, same, codissent int
}

// Tracker accumulates evidence about nodes across requests. Feed it every
// request's responses and the answer the cluster settled on; ask it whom to
// trust. Safe for concurrent use.
type Tracker struct {
	cfg Config

	mu    sync.Mutex
	n     int
	nodes map[string]*nodeState
	pairs map[[2]string]*pairState
}

func NewTracker(cfg Config) *Tracker {
	return &Tracker{cfg: cfg, nodes: map[string]*nodeState{}, pairs: map[[2]string]*pairState{}}
}

// Observe folds in one request: what every node answered and the consensus
// the cluster adopted. Cancelled responses are ignored; failures count as
// disagreement.
func (t *Tracker) Observe(resps []cluster.Response, consensus []int32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.n++

	var answered []cluster.Response
	for _, r := range resps {
		if r.Err == cluster.ErrCancelled {
			continue
		}
		s := t.nodes[r.Node]
		if s == nil {
			s = &nodeState{base: 0.2}
			t.nodes[r.Node] = s
		}
		x := 1.0 // dissent indicator
		if r.OK() {
			answered = append(answered, r)
			if cluster.Equal(r.Tokens, consensus) {
				x = 0
			}
		}
		s.seen++
		s.agree = t.cfg.Forget*s.agree + (1 - x)
		s.dissent = t.cfg.Forget*s.dissent + x

		// One-sided CUSUM on the dissent indicator, against the node's
		// own baseline: honest nodes of any quality hover near zero,
		// a node that changes behaviour climbs by ~1 per request.
		//
		// The baseline is learnt slowly and unconditionally. (Learning it
		// only while the statistic is at rest sounds safer but samples
		// mostly agreements, biasing the baseline low and raising false
		// alarms on honest-but-lossy nodes.) A node that stays bad
		// eventually becomes its own baseline and the alarm clears — by
		// then its trust has collapsed, which is the steady-state signal.
		s.cusum = min(2*t.cfg.DriftAlarm, max(0, s.cusum+x-s.base-t.cfg.DriftSlack))
		s.base += 0.01 * (x - s.base)
		switch {
		case s.cusum >= t.cfg.DriftAlarm:
			s.drifted = true
		case s.cusum == 0:
			s.drifted = false
		}
	}

	for i, a := range answered {
		for _, b := range answered[i+1:] {
			k := [2]string{a.Node, b.Node}
			if k[0] > k[1] {
				k[0], k[1] = k[1], k[0]
			}
			p := t.pairs[k]
			if p == nil {
				p = &pairState{}
				t.pairs[k] = p
			}
			p.joint++
			if cluster.Equal(a.Tokens, b.Tokens) {
				p.same++
				if !cluster.Equal(a.Tokens, consensus) {
					p.codissent++
				}
			}
		}
	}

	bloc := t.blocOf()
	for id, s := range t.nodes { // stamp first detection, for time-to-detect
		if s.suspectAt == 0 && len(t.reasons(id, s, bloc)) > 0 {
			s.suspectAt = t.n
		}
	}
}

func (t *Tracker) trust(s *nodeState) float64 {
	return (s.agree + 1) / (s.agree + s.dissent + 2) // Laplace-smoothed: 0.5 for a stranger
}

// Trust is the tracker's current estimate of how often the node agrees
// with the consensus, discounted towards recent behaviour. Unknown nodes
// get 0.5.
func (t *Tracker) Trust(node string) float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	if s := t.nodes[node]; s != nil {
		return t.trust(s)
	}
	return 0.5
}

// blocs returns groups of nodes that repeatedly deviate from the consensus
// with identical answers. Two honest nodes do this occasionally — they can
// take the same wrong turn — but not on a tenth of their shared requests.
func (t *Tracker) blocs() [][]string {
	parent := map[string]string{}
	var find func(string) string
	find = func(x string) string {
		if parent[x] == "" || parent[x] == x {
			parent[x] = x
			return x
		}
		parent[x] = find(parent[x])
		return parent[x]
	}
	for k, p := range t.pairs {
		if p.joint >= t.cfg.MinJoint && float64(p.codissent) >= t.cfg.CoDissent*float64(p.joint) {
			parent[find(k[0])] = find(k[1])
		}
	}
	groups := map[string][]string{}
	for id := range parent {
		root := find(id)
		groups[root] = append(groups[root], id)
	}
	var out [][]string
	for _, g := range groups {
		if len(g) > 1 {
			sort.Strings(g)
			out = append(out, g)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i][0] < out[j][0] })
	return out
}

func (t *Tracker) blocOf() map[string]int {
	of := map[string]int{}
	for i, g := range t.blocs() {
		for _, id := range g {
			of[id] = i + 1
		}
	}
	return of
}

// Blocs lists the colluding groups found so far.
func (t *Tracker) Blocs() [][]string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.blocs()
}

func (t *Tracker) reasons(id string, s *nodeState, bloc map[string]int) []string {
	var why []string
	if bloc[id] > 0 {
		why = append(why, Collusion)
	}
	if s.drifted {
		why = append(why, Drift)
	}
	if s.seen >= t.cfg.MinSeen && t.trust(s) < t.cfg.TrustFloor {
		why = append(why, Untrusted)
	}
	return why
}

// Suspects lists the nodes the tracker currently considers untrustworthy,
// sorted by node.
func (t *Tracker) Suspects() []Suspect {
	t.mu.Lock()
	defer t.mu.Unlock()
	bloc := t.blocOf()
	var out []Suspect
	for id, s := range t.nodes {
		if why := t.reasons(id, s, bloc); len(why) > 0 {
			out = append(out, Suspect{Node: id, Reasons: why, Trust: t.trust(s), DetectedAt: s.suspectAt})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Node < out[j].Node })
	return out
}

// Skip records that a request went by without being used as evidence, so
// DetectedAt stays in units of requests seen rather than requests learnt
// from.
func (t *Tracker) Skip() {
	t.mu.Lock()
	t.n++
	t.mu.Unlock()
}

// Observed is the number of requests folded in so far.
func (t *Tracker) Observed() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.n
}
