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
	// Ablations of adaptive, for measuring what each defence contributes.
	for name, o := range map[string]AdaptiveOptions{
		"adaptive-no-screen": {NoScreen: true},
		"adaptive-no-trust":  {NoTrust: true},
		"adaptive-no-blocs":  {NoBlocs: true},
		"adaptive-unbounded": {Unbounded: true},
		// The naive design: learn from everything, with nothing bounded.
		"adaptive-naive": {Unbounded: true, LearnAlways: true},
	} {
		registry[name] = func() cluster.Aggregator {
			a := AdaptiveWith(o).(*adaptive)
			a.name = name
			return a
		}
	}
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
//   - Per-request screening: answers flagged as outliers, over-confident,
//     implausibly fast or truncated are set aside before the vote. This
//     uses only the request in hand, so it cannot feed on itself.
//   - A learned discount for nodes with a worse record than the typical
//     voter, or whose disagreement has just jumped (drift). A good record
//     earns an ordinary vote, never a bigger one, so a sleeper's patience
//     buys nothing.
//   - Bloc collapsing: nodes found to deviate in lockstep share one vote
//     when they answer identically.
//
// The last two learn from the strategy's own decisions, and anything that
// does is bistable: if the adversaries win often enough early on, it is
// the honest nodes whose record looks bad, and the strategy talks itself
// into trusting the wrong side for good. Four limits prevent that:
//
//   - Learn only from decisive requests: those a strict majority of the
//     answering nodes settled outright. A minority cannot manufacture
//     one, so what it wins never becomes evidence.
//   - A head-count majority is final. At every position, a token that
//     more than half of the answering nodes back is taken whatever the
//     learned weights say; a screened answer is not a backer. Discounts
//     and shared votes decide only what no majority settles, so honest
//     nodes that agree cannot be out-voted by anything the strategy has
//     learned, and a captured state cannot sustain itself.
//   - Bounded influence. With n voters, a learned discount can lower a
//     vote to n/(n+1) of a full one and no further. That is enough to
//     settle any contest between equal head counts in favour of the side
//     with the better record, and not enough to let a smaller group
//     out-vote a larger one on its record alone.
//   - Only minority blocs collapse. Honest nodes that are right also
//     deviate together whenever the adversaries win a vote; a bloc of at
//     least half the voters is taken to be exactly that, and left alone.
//
// The second limit is what makes the others safe to combine, and it has a
// price. Sharing a vote is not bounded the way a discount is: honest
// replicas that share their mistakes (two copies of one quantised model,
// say) are a bloc as far as the evidence goes, and one discounted vote
// between them hands a 4-against-3 contest to the three. Any ordering that
// lets learned state overrule a majority can be captured that way —
// ranking by independent evidence first only moves the threshold to three
// such replicas. The price is that a majority is taken at its word even
// when part of it is a bloc the strategy has already identified:
// colluders joined by an honest node making the same mistake win that
// request, where trusting the inference would have stopped them.
func Adaptive() cluster.Aggregator { return AdaptiveWith(AdaptiveOptions{}) }

// AdaptiveOptions switches off parts of the adaptive strategy, for
// ablation: which defence is earning its keep?
type AdaptiveOptions struct {
	NoScreen    bool // skip per-request screening
	NoTrust     bool // skip the learned discount (trust and drift)
	NoBlocs     bool // skip bloc collapsing
	Unbounded   bool // let learned discounts go to zero, collapse any bloc, and let weights overrule a majority
	LearnAlways bool // learn from every decision, not just the decisive ones
}

// AdaptiveWith builds the adaptive strategy with some defences disabled.
func AdaptiveWith(o AdaptiveOptions) cluster.Aggregator {
	cfg := detect.Default()
	return &adaptive{name: "adaptive", cfg: cfg, tr: detect.NewTracker(cfg), opt: o}
}

type adaptive struct {
	name string
	cfg  detect.Config
	tr   *detect.Tracker
	opt  AdaptiveOptions
}

func (a *adaptive) Name() string              { return a.name }
func (a *adaptive) Trust(node string) float64 { return a.tr.Trust(node) }
func (a *adaptive) Tracker() *detect.Tracker  { return a.tr }

func (a *adaptive) Aggregate(_ cluster.Request, resps []cluster.Response) cluster.Decision {
	answered := cluster.Answered(resps)

	// Screen this request's answers.
	screened := map[string]bool{}
	if !a.opt.NoScreen {
		for _, f := range a.cfg.Inspect(resps) {
			if f.Reason != detect.Failed && detect.Adversarial(f.Reason) {
				screened[f.Node] = true
			}
		}
		if len(screened) >= len(answered) { // never screen out everyone
			screened = map[string]bool{}
		}
	}
	byNode := map[string]cluster.Response{}
	trust := map[string]float64{}
	var voting []float64
	for _, r := range answered {
		byNode[r.Node] = r
		trust[r.Node] = a.tr.Trust(r.Node)
		if !screened[r.Node] {
			voting = append(voting, trust[r.Node])
		}
	}
	voters := float64(len(voting))
	typical := cluster.Median(voting)

	// The learned discount: 0 for a node with at least a typical record,
	// rising to 1 for one with no record of agreeing, or one that has
	// just changed behaviour.
	discount := map[string]float64{}
	if !a.opt.NoTrust {
		for id, t := range trust {
			if typical > 0 && t < typical {
				discount[id] = 1 - t/typical
			}
		}
		for _, s := range a.tr.Suspects() {
			for _, why := range s.Reasons {
				if why == detect.Drift {
					discount[s.Node] = 1
				}
			}
		}
	}
	floor := voters / (voters + 1) // the least a learned discount may leave of a vote
	if a.opt.Unbounded {
		floor = 0
	}

	// Members of a minority bloc who give the same answer split one vote.
	share := map[string]float64{}
	if !a.opt.NoBlocs {
		for _, bloc := range a.tr.Blocs() {
			present := 0
			same := map[string][]string{}
			for _, id := range bloc {
				if r, ok := byNode[id]; ok && !screened[id] {
					present++
					k := cluster.Key(r.Tokens)
					same[k] = append(same[k], id)
				}
			}
			// Measured against everyone who answered, not just those who
			// survived screening: screening a few honest stragglers must
			// not promote a colluding minority to "half the cluster".
			if !a.opt.Unbounded && 2*present >= len(answered) {
				continue // at least half the nodes: presumed to be the honest majority
			}
			for _, ids := range same {
				for _, id := range ids {
					share[id] = 1 / float64(len(ids))
				}
			}
		}
	}

	weight := func(r cluster.Response) float64 {
		if screened[r.Node] {
			return 0
		}
		w := 1 - (1-floor)*math.Min(1, discount[r.Node])
		if s, ok := share[r.Node]; ok {
			w *= s
		}
		return w
	}
	// A head-count majority is final: discounts and shared votes only
	// decide what no majority settles.
	d := prefixVoteGuarded(resps, weight, !a.opt.Unbounded)

	// Learn only from requests that a strict majority of the answering
	// nodes settled outright. Fewer than half the nodes cannot produce
	// such a result on their own, so nothing an adversarial minority wins
	// — a tie broken its way, a bloc that out-voted split honest nodes —
	// ever becomes evidence. Learning from contested decisions is what
	// lets a strategy talk itself into trusting the wrong side.
	heads, _ := tally(resps, one)
	decisive := len(heads) > 0 && 2*len(heads[0].nodes) > len(answered) && cluster.Equal(heads[0].tokens, d.Tokens)
	if decisive || a.opt.LearnAlways {
		a.tr.Observe(resps, d.Tokens)
	} else {
		a.tr.Skip()
	}

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
