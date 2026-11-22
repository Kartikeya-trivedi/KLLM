// Package detect finds responses and nodes that should not be trusted,
// using only what an orchestrator can observe.
//
// It works at two time scales. Inspect looks at one request: among the
// answers to the same prompt, which are unlike the rest, suspiciously
// certain, too fast, too short, or missing? Tracker looks across requests:
// which nodes keep losing the vote, which changed behaviour (a sleeper
// waking), and which deviate *together* (a colluding bloc)?
//
// Nothing here knows the right answer. Every signal is relative to the
// other nodes, which is the method's strength and its limit: once the
// adversaries are the consensus, it is the honest nodes that look deviant.
package detect

import (
	"sort"

	"kllm/cluster"
)

// Reasons a response is flagged.
const (
	Failed        = "failed"        // error or timeout
	Outlier       = "outlier"       // far less similar to the other answers than they are to each other
	Minority      = "minority"      // disagrees with an answer most nodes gave
	Overconfident = "overconfident" // reports far more confidence than its peers
	Fast          = "fast"          // answered much sooner than its peers could
	Slow          = "slow"          // answered much later than its peers
	Truncated     = "truncated"     // much shorter than its peers' answers
)

// Flag marks one response as anomalous.
type Flag struct {
	Node   string  `json:"node"`
	Reason string  `json:"reason"`
	Score  float64 `json:"score"` // how extreme, in the units of the test that fired
}

// Config holds every threshold. The zero value is not useful; start from
// Default.
type Config struct {
	// Inspect: one request.
	MinPeers     int     // answers needed before peers are compared
	SimZ         float64 // outlier: robust z of mean similarity below -SimZ ...
	SimGap       float64 // ... and at least this far below the median similarity
	ConfZ        float64 // overconfident: robust z of reported confidence above ConfZ ...
	ConfGap      float64 // ... and at least this far above the median confidence
	LatencyZ     float64 // fast/slow: robust z of latency beyond +/-LatencyZ ...
	LatencyRatio float64 // ... and at least this factor away from the median
	TruncFrac    float64 // truncated: shorter than this fraction of the median length

	// Tracker: across requests.
	Forget     float64 // per-request decay of a node's agreement history (window ~ 1/(1-Forget))
	TrustFloor float64 // a node trusted less than this is a suspect ...
	MinSeen    int     // ... once it has answered this many requests
	DriftSlack float64 // CUSUM: disagreement tolerated above a node's own baseline
	DriftAlarm float64 // CUSUM: accumulated excess that raises the alarm
	MinJoint   int     // collusion: requests two nodes must share before they are compared
	CoDissent  float64 // collusion: fraction of shared requests on which both deviate identically
}

// Default is the configuration the experiments use.
func Default() Config {
	return Config{
		MinPeers: 3,
		SimZ:     3, SimGap: 0.25,
		ConfZ: 3, ConfGap: 0.05,
		LatencyZ: 4, LatencyRatio: 2,
		TruncFrac: 0.75,

		Forget: 0.98, TrustFloor: 0.5, MinSeen: 20,
		DriftSlack: 0.25, DriftAlarm: 6,
		MinJoint: 30, CoDissent: 0.10,
	}
}

// Adversarial reports whether a flag is evidence of misbehaviour rather
// than of an ordinary mistake or a slow machine. Minority and Slow are
// what honest nodes do on a bad day; the rest are not.
func Adversarial(reason string) bool {
	return reason != Minority && reason != Slow
}

// Inspect examines the responses to one request and flags the anomalous
// ones. A response can carry several flags. Flags come back sorted by node,
// then reason.
func (c Config) Inspect(resps []cluster.Response) []Flag {
	var flags []Flag
	answered := make([]cluster.Response, 0, len(resps))
	for _, r := range resps {
		if r.OK() {
			answered = append(answered, r)
		} else if r.Err != cluster.ErrCancelled {
			flags = append(flags, Flag{r.Node, Failed, 1})
		}
	}
	// Work in a fixed order, not the order the responses arrived in. The
	// tests below compare sums of similarities against thresholds, and a
	// sum taken in a different order rounds differently — enough to move a
	// borderline answer across one.
	sort.SliceStable(answered, func(i, j int) bool { return answered[i].Node < answered[j].Node })
	if len(answered) >= c.MinPeers {
		n := len(answered)
		sims := make([]float64, n)  // mean similarity to the other answers
		confs := make([]float64, n) // reported sequence confidence
		lats := make([]float64, n)
		lens := make([]float64, n)
		var reported []float64
		counts := map[string]int{}
		terms := make([]float64, 0, n)
		for i, r := range answered {
			// Add each node's similarities smallest first. Two nodes that
			// gave the same answer have the same similarities to everyone
			// else, met in a different order; summed as they come, their
			// means can differ in the last bit and land on opposite sides
			// of a threshold.
			terms = terms[:0]
			for j, o := range answered {
				if i != j {
					terms = append(terms, cluster.Similarity(r.Tokens, o.Tokens))
				}
			}
			sort.Float64s(terms)
			for _, s := range terms {
				sims[i] += s
			}
			sims[i] /= float64(n - 1)
			if p, ok := cluster.SeqConf(r.Conf); ok {
				confs[i] = p
				reported = append(reported, p)
			} else {
				confs[i] = -1
			}
			lats[i], lens[i] = r.LatencyMs, float64(len(r.Tokens))
			counts[cluster.Key(r.Tokens)]++
		}
		var top []int32
		topCount := 0
		for _, r := range answered {
			if k := counts[cluster.Key(r.Tokens)]; k > topCount {
				top, topCount = r.Tokens, k
			}
		}
		medSim, medLat, medLen := cluster.Median(sims), cluster.Median(lats), cluster.Median(lens)
		medConf := cluster.Median(reported)

		for i, r := range answered {
			// The gaps that can occur are a lattice (multiples of 1/96 for
			// seven nodes and 16 tokens) and the threshold sits on it, so
			// compare with a tolerance: a gap of exactly SimGap reaches it.
			if gap := medSim - sims[i]; gap >= c.SimGap-1e-9 && cluster.RobustZ(sims[i], sims) <= -c.SimZ {
				flags = append(flags, Flag{r.Node, Outlier, gap})
			}
			if 2*topCount > n && !cluster.Equal(r.Tokens, top) {
				flags = append(flags, Flag{r.Node, Minority, 1 - cluster.Similarity(r.Tokens, top)})
			}
			if len(reported) >= c.MinPeers && confs[i] >= 0 {
				if gap := confs[i] - medConf; gap >= c.ConfGap && cluster.RobustZ(confs[i], reported) >= c.ConfZ {
					flags = append(flags, Flag{r.Node, Overconfident, gap})
				}
			}
			if medLat > 0 && lats[i] > 0 {
				switch z := cluster.RobustZ(lats[i], lats); {
				case z <= -c.LatencyZ && lats[i]*c.LatencyRatio <= medLat:
					flags = append(flags, Flag{r.Node, Fast, medLat / lats[i]})
				case z >= c.LatencyZ && lats[i] >= medLat*c.LatencyRatio:
					flags = append(flags, Flag{r.Node, Slow, lats[i] / medLat})
				}
			}
			if medLen > 0 && lens[i] < c.TruncFrac*medLen {
				flags = append(flags, Flag{r.Node, Truncated, 1 - lens[i]/medLen})
			}
		}
	}
	sort.Slice(flags, func(i, j int) bool {
		if flags[i].Node != flags[j].Node {
			return flags[i].Node < flags[j].Node
		}
		return flags[i].Reason < flags[j].Reason
	})
	return flags
}
