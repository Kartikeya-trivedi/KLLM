package aggregate

import (
	"math"
	"sort"

	"kllm/cluster"
	"kllm/cluster/detect"
)

func init() {
	registry["reputation"] = Reputation
	registry["adaptive"] = Adaptive
}

// The adaptive strategies carry state across requests: they watch which
// nodes agree with the answers the cluster settles on and weight later
// votes accordingly. Both learn from their own decisions — there is no
// ground truth online — so they are only as good as the consensus they
// bootstrap from. A fresh instance must be used per run.

// Reputation is trust-weighted voting, the textbook adaptive scheme: each
// node's vote counts in proportion to how often it has agreed with past
// decisions. It suppresses nodes that are persistently wrong. It also
// rewards whoever has been agreeable so far, which is precisely what a
// sleeper is: when trusted nodes turn, they out-vote the honest ones until
// enough lost votes have worn their record down.
func Reputation() cluster.Aggregator {
	return &reputation{tr: detect.NewTracker(detect.Default())}
}

type reputation struct{ tr *detect.Tracker }

func (a *reputation) Name() string              { return "reputation" }
func (a *reputation) Trust(node string) float64 { return a.tr.Trust(node) }
func (a *reputation) Tracker() *detect.Tracker  { return a.tr }

func (a *reputation) Aggregate(_ cluster.Request, resps []cluster.Response) cluster.Decision {
	d := decide(tally(resps, func(r cluster.Response) float64 { return a.tr.Trust(r.Node) }))
	a.tr.Observe(resps, d.Tokens)
	return d
}

// Adaptive combines the defences that held up individually:
//
//   - Token-level voting, so honest nodes that diverge late still pool
//     their support.
//   - Trust weights that can only discount: a node's weight is its trust,
//     capped at the median trust of the nodes answering. A good record
//     earns an ordinary vote, never a bigger one, so a sleeper's patience
//     buys nothing.
//   - Per-request screening: answers flagged as outliers, over-confident,
//     implausibly fast or truncated are set aside before the vote.
//   - Bloc collapsing: nodes found to deviate in lockstep share one vote
//     when they answer identically, however many of them there are.
//   - Drift: a node whose disagreement jumps is discounted at once, without
//     waiting for its trust to decay.
func Adaptive() cluster.Aggregator {
	cfg := detect.Default()
	return &adaptive{cfg: cfg, tr: detect.NewTracker(cfg)}
}

type adaptive struct {
	cfg detect.Config
	tr  *detect.Tracker
}

// driftPenalty scales the vote of a node whose behaviour has changed.
const driftPenalty = 0.1

func (a *adaptive) Name() string              { return "adaptive" }
func (a *adaptive) Trust(node string) float64 { return a.tr.Trust(node) }
func (a *adaptive) Tracker() *detect.Tracker  { return a.tr }

func (a *adaptive) Aggregate(_ cluster.Request, resps []cluster.Response) cluster.Decision {
	answered := cluster.Answered(resps)

	// Screen this request's answers.
	screened := map[string]bool{}
	for _, f := range a.cfg.Inspect(resps) {
		if f.Reason != detect.Failed && detect.Adversarial(f.Reason) {
			screened[f.Node] = true
		}
	}
	if len(screened) >= len(answered) { // never screen out everyone
		screened = map[string]bool{}
	}

	// Trust, capped at the median of those voting.
	trust := map[string]float64{}
	var voting []float64
	for _, r := range answered {
		trust[r.Node] = a.tr.Trust(r.Node)
		if !screened[r.Node] {
			voting = append(voting, trust[r.Node])
		}
	}
	ceiling := cluster.Median(voting)

	drifted := map[string]bool{}
	for _, s := range a.tr.Suspects() {
		for _, why := range s.Reasons {
			if why == detect.Drift {
				drifted[s.Node] = true
			}
		}
	}

	// Members of a bloc who give the same answer split one vote.
	share := map[string]float64{}
	byNode := map[string]cluster.Response{}
	for _, r := range answered {
		byNode[r.Node] = r
	}
	for _, bloc := range a.tr.Blocs() {
		same := map[string][]string{}
		for _, id := range bloc {
			if r, ok := byNode[id]; ok && !screened[id] {
				k := cluster.Key(r.Tokens)
				same[k] = append(same[k], id)
			}
		}
		for _, ids := range same {
			for _, id := range ids {
				share[id] = 1 / float64(len(ids))
			}
		}
	}

	weight := func(r cluster.Response) float64 {
		if screened[r.Node] {
			return 0
		}
		w := math.Min(trust[r.Node], ceiling)
		if drifted[r.Node] {
			w *= driftPenalty
		}
		if s, ok := share[r.Node]; ok {
			w *= s
		}
		return w
	}
	d := prefixVote(resps, weight)
	a.tr.Observe(resps, d.Tokens)

	// Report whom this decision distrusted: screened answers, plus known
	// suspects that dissented from it.
	suspects := map[string]bool{}
	for id := range screened {
		suspects[id] = true
	}
	for _, s := range a.tr.Suspects() {
		if r, ok := byNode[s.Node]; ok && !cluster.Equal(r.Tokens, d.Tokens) {
			suspects[s.Node] = true
		}
	}
	for id := range suspects {
		d.Suspects = append(d.Suspects, id)
	}
	sort.Strings(d.Suspects)
	return d
}
