// Package aggregate combines the answers of several inference nodes into
// one. Every strategy implements cluster.Aggregator and sees only what an
// orchestrator can observe: each node's tokens, the confidences it chose to
// report, and how long it took.
//
// The stateless families:
//
//	voting      majority, quorum      identical answers pool their votes
//	token-level token-vote            votes pool on shared prefixes
//	ranking     mbr, borda            answers are ranked by agreement with the rest
//	confidence  confidence,           votes are weighted by self-reported
//	            confidence-clipped,   confidence
//	            max-confidence
//	baseline    single                no aggregation: one node's answer
//
// The adaptive strategies, which learn whom to trust across requests, are
// in adaptive.go.
package aggregate

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"kllm/cluster"
)

// stateless adapts a function to cluster.Aggregator.
type stateless struct {
	name string
	fn   func(req cluster.Request, resps []cluster.Response) cluster.Decision
}

func (s stateless) Name() string { return s.name }
func (s stateless) Aggregate(req cluster.Request, resps []cluster.Response) cluster.Decision {
	return s.fn(req, resps)
}

var registry = map[string]func() cluster.Aggregator{
	"single":             Single,
	"majority":           Majority,
	"quorum":             func() cluster.Aggregator { return Quorum(0.5) },
	"token-vote":         TokenVote,
	"mbr":                MBR,
	"borda":              Borda,
	"confidence":         Confidence,
	"confidence-clipped": ConfidenceClipped,
	"max-confidence":     MaxConfidence,
}

// New returns a fresh instance of the named strategy.
func New(name string) (cluster.Aggregator, error) {
	if mk, ok := registry[name]; ok {
		return mk(), nil
	}
	return nil, fmt.Errorf("aggregate: unknown strategy %q (have %s)", name, strings.Join(Names(), ", "))
}

// Names lists the registered strategies.
func Names() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// --- shared machinery ------------------------------------------------------

// bucket is one distinct answer and the nodes behind it.
type bucket struct {
	key    string
	tokens []int32
	nodes  []string
	weight float64
	sim    float64 // mean similarity to the other distinct answers, for tie-breaks
}

const eps = 1e-12

// tally groups answers by exact content and orders them best first: by
// weight, then by head count, then by resemblance to the other answers,
// then by content — so the outcome never depends on the order responses
// arrived in.
func tally(resps []cluster.Response, weight func(cluster.Response) float64) (buckets []*bucket, total float64) {
	byKey := map[string]*bucket{}
	for _, r := range resps {
		if !r.OK() {
			continue
		}
		k := cluster.Key(r.Tokens)
		b := byKey[k]
		if b == nil {
			b = &bucket{key: k, tokens: r.Tokens}
			byKey[k] = b
			buckets = append(buckets, b)
		}
		w := math.Max(0, weight(r))
		b.nodes = append(b.nodes, r.Node)
		b.weight += w
		total += w
	}
	for _, b := range buckets {
		sort.Strings(b.nodes)
		for _, o := range buckets {
			if o != b {
				b.sim += cluster.Similarity(b.tokens, o.tokens) / float64(len(buckets)-1)
			}
		}
	}
	sort.Slice(buckets, func(i, j int) bool {
		a, b := buckets[i], buckets[j]
		switch {
		case math.Abs(a.weight-b.weight) > eps:
			return a.weight > b.weight
		case len(a.nodes) != len(b.nodes):
			return len(a.nodes) > len(b.nodes)
		case math.Abs(a.sim-b.sim) > eps:
			return a.sim > b.sim
		}
		return a.key < b.key
	})
	return
}

// decide turns an ordered tally into a Decision.
func decide(buckets []*bucket, total float64) cluster.Decision {
	if len(buckets) == 0 {
		return cluster.Decision{Abstained: true}
	}
	if total <= 0 { // every vote weighed nothing: fall back to counting heads
		return decide(tally(asResponses(buckets), one))
	}
	d := cluster.Decision{Tokens: buckets[0].tokens, Winners: buckets[0].nodes, Share: buckets[0].weight / total}
	d.Margin = d.Share
	if len(buckets) > 1 {
		d.Margin = (buckets[0].weight - buckets[1].weight) / total
	}
	return d
}

func asResponses(buckets []*bucket) (out []cluster.Response) {
	for _, b := range buckets {
		for _, n := range b.nodes {
			out = append(out, cluster.Response{Node: n, Tokens: b.tokens})
		}
	}
	return
}

func one(cluster.Response) float64 { return 1 }

// --- voting ----------------------------------------------------------------

// Majority is plurality voting over whole answers: the most common answer
// wins. It is the classic defence and works while the honest nodes that
// got it right outnumber any group agreeing on something else — which is a
// stronger condition than "honest nodes outnumber adversaries", because
// honest nodes that err split their votes.
func Majority() cluster.Aggregator {
	return stateless{"majority", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		return decide(tally(resps, one))
	}}
}

// Quorum is majority voting that refuses to answer unless more than frac of
// the nodes it heard from — including those that failed — back the winner.
// It trades coverage for confidence: an abstention is a visible failure
// where plain majority would return a possibly-wrong answer.
func Quorum(frac float64) cluster.Aggregator {
	return stateless{"quorum", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		buckets, _ := tally(resps, one)
		d := decide(buckets, float64(len(resps)))
		if d.Share <= frac {
			d.Abstained = true
		}
		return d
	}}
}

// Single is no aggregation at all: the answer of one node, picked by
// request hash. It is the baseline every strategy has to beat, and the
// accuracy a deployment without redundancy gets.
func Single() cluster.Aggregator {
	return stateless{"single", func(req cluster.Request, resps []cluster.Response) cluster.Decision {
		ids := make([]string, len(resps))
		byNode := map[string]cluster.Response{}
		for i, r := range resps {
			ids[i], byNode[r.Node] = r.Node, r
		}
		sort.Strings(ids)
		for _, id := range cluster.Order(cluster.Policy{}, req, ids, nil) {
			if r := byNode[id]; r.OK() {
				return cluster.Decision{Tokens: r.Tokens, Winners: []string{id}, Share: 1, Margin: 1}
			}
		}
		return cluster.Decision{Abstained: true}
	}}
}

// --- token-level voting ----------------------------------------------------

// TokenVote votes one token at a time, but only among the answers that
// agree with everything chosen so far. Whole-answer voting wastes the
// votes of honest nodes that diverged late and in different directions;
// here they still back the prefix they share, so a group agreeing on a
// fabricated answer has to outnumber everyone who is right *up to the
// point it departs*, not just the nodes that are right to the end.
//
// The walk always ends on an answer some node actually gave: it never
// splices tokens from decodes that had diverged, which would produce text
// no model generated.
func TokenVote() cluster.Aggregator {
	return stateless{"token-vote", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		return prefixVote(resps, one)
	}}
}

const endOfAnswer = int64(math.MinInt64) // sentinel "token": the answer stops here

func prefixVote(resps []cluster.Response, weight func(cluster.Response) float64) cluster.Decision {
	type voter struct {
		r cluster.Response
		w float64
	}
	var alive []voter
	var total float64
	for _, r := range resps {
		if r.OK() {
			w := math.Max(0, weight(r))
			alive = append(alive, voter{r, w})
			total += w
		}
	}
	if len(alive) == 0 {
		return cluster.Decision{Abstained: true}
	}
	if total <= 0 {
		return prefixVote(resps, one)
	}
	d := cluster.Decision{Margin: 1}
	for pos := 0; ; pos++ {
		type option struct {
			tok   int64
			w     float64
			heads int
		}
		opts := map[int64]*option{}
		for _, v := range alive {
			tok := endOfAnswer
			if pos < len(v.r.Tokens) {
				tok = int64(v.r.Tokens[pos])
			}
			o := opts[tok]
			if o == nil {
				o = &option{tok: tok}
				opts[tok] = o
			}
			o.w += v.w
			o.heads++
		}
		ranked := make([]*option, 0, len(opts))
		for _, o := range opts {
			ranked = append(ranked, o)
		}
		sort.Slice(ranked, func(i, j int) bool {
			a, b := ranked[i], ranked[j]
			switch {
			case math.Abs(a.w-b.w) > eps:
				return a.w > b.w
			case a.heads != b.heads:
				return a.heads > b.heads
			}
			return a.tok < b.tok
		})
		if len(ranked) > 1 { // the tightest contest along the path
			d.Margin = math.Min(d.Margin, (ranked[0].w-ranked[1].w)/total)
		}
		best := ranked[0]
		kept := alive[:0:0]
		for _, v := range alive {
			if (pos < len(v.r.Tokens) && int64(v.r.Tokens[pos]) == best.tok) ||
				(pos >= len(v.r.Tokens) && best.tok == endOfAnswer) {
				kept = append(kept, v)
			}
		}
		alive = kept
		if best.tok == endOfAnswer {
			break
		}
	}
	d.Tokens = alive[0].r.Tokens
	for _, v := range alive {
		d.Winners = append(d.Winners, v.r.Node)
		d.Share += v.w / total
	}
	sort.Strings(d.Winners)
	return d
}

// --- ranking ---------------------------------------------------------------

// MBR picks the answer with the highest total similarity to all the other
// answers — minimum Bayes risk decoding with the other nodes as the sample
// of plausible outputs. Where voting counts only exact matches, this
// counts near-matches in proportion, so it degrades gracefully when no two
// honest answers are identical.
func MBR() cluster.Aggregator {
	return stateless{"mbr", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		return rank(resps, func(a []cluster.Response) []float64 {
			scores := make([]float64, len(a))
			for i := range a {
				for j := range a {
					if i != j {
						scores[i] += cluster.Similarity(a[i].Tokens, a[j].Tokens)
					}
				}
			}
			return scores
		})
	}}
}

// Borda is rank aggregation: every node ranks the distinct answers by how
// closely they resemble its own, an answer earns a point for each distinct
// answer it finishes above on a ballot, and the most points wins. It uses
// only the order of similarities, not their size — which also means a node
// gives its own answer the maximum regardless of how implausible it is, so
// a bloc of identical ballots carries real weight.
func Borda() cluster.Aggregator {
	return stateless{"borda", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		return rank(resps, func(a []cluster.Response) []float64 {
			// The candidates are the distinct answers, not the responses:
			// counting responses beaten would reward an answer for being
			// in the minority.
			cand := map[string]int{}
			var distinct [][]int32
			for _, r := range a {
				if k := cluster.Key(r.Tokens); cand[k] == 0 {
					distinct = append(distinct, r.Tokens)
					cand[k] = len(distinct)
				}
			}
			points := make([]float64, len(distinct))
			sims := make([]float64, len(distinct))
			for _, voter := range a {
				for i, c := range distinct {
					sims[i] = cluster.Similarity(voter.Tokens, c)
				}
				for i := range distinct {
					for j := range distinct {
						if sims[i] > sims[j]+eps {
							points[i]++
						}
					}
				}
			}
			scores := make([]float64, len(a))
			for i, r := range a {
				scores[i] = points[cand[cluster.Key(r.Tokens)]-1]
			}
			return scores
		})
	}}
}

// rank scores each answered response and returns the best-scored answer.
// Identical answers score identically, so the decision is still a choice
// between distinct answers; Share and Margin are reported in score space.
func rank(resps []cluster.Response, score func([]cluster.Response) []float64) cluster.Decision {
	answered := cluster.Answered(resps)
	if len(answered) == 0 {
		return cluster.Decision{Abstained: true}
	}
	scores := score(answered)
	byNode := make(map[string]float64, len(answered))
	for i, r := range answered {
		byNode[r.Node] = scores[i]
	}
	buckets, _ := tally(answered, func(r cluster.Response) float64 { return byNode[r.Node] })
	var total float64
	for _, b := range buckets { // one score per distinct answer, not per supporter
		b.weight /= float64(len(b.nodes))
		total += b.weight
	}
	sort.SliceStable(buckets, func(i, j int) bool {
		a, b := buckets[i], buckets[j]
		if math.Abs(a.weight-b.weight) > eps {
			return a.weight > b.weight
		}
		return len(a.nodes) > len(b.nodes)
	})
	if total <= 0 { // nobody resembles anybody: nothing to rank on
		return decide(tally(answered, one))
	}
	return decide(buckets, total)
}

// --- confidence ------------------------------------------------------------

// confWeights gives each answering node its vote weight under confidence
// weighting: the geometric-mean token probability it reported. A node that
// reports nothing gets the typical confidence of those that did.
func confWeights(resps []cluster.Response) map[string]float64 {
	w := map[string]float64{}
	var reported []float64
	for _, r := range resps {
		if !r.OK() {
			continue
		}
		if c, ok := cluster.SeqConf(r.Conf); ok {
			w[r.Node] = c
			reported = append(reported, c)
		}
	}
	typical := 0.5
	if len(reported) > 0 {
		typical = cluster.Median(reported)
	}
	for _, r := range resps {
		if _, ok := w[r.Node]; r.OK() && !ok {
			w[r.Node] = typical
		}
	}
	return w
}

// Confidence weights each vote by the confidence its node reported. On
// honest replicas this is a real gain: a replica that took a low-margin
// wrong turn says so. Its weakness is that the weight is self-reported.
func Confidence() cluster.Aggregator {
	return stateless{"confidence", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		w := confWeights(resps)
		return decide(tally(resps, func(r cluster.Response) float64 { return w[r.Node] }))
	}}
}

// ConfidenceClipped is confidence weighting in which no vote may weigh more
// than the median reported confidence. Reporting low confidence still
// discounts a vote, but claiming certainty buys nothing beyond an ordinary
// vote, so a node cannot inflate its way to a win.
//
// The cost is real: with the ceiling at the median, a confident minority
// can no longer overrule an unsure majority, so on honest replicas this
// keeps only part of what plain confidence weighting offers. And a liar
// sitting exactly at the ceiling still out-weighs an honest node that is
// unsure, so under spoofing it lands near majority voting, not above it.
func ConfidenceClipped() cluster.Aggregator {
	return stateless{"confidence-clipped", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		w := confWeights(resps)
		vals := make([]float64, 0, len(w))
		for _, c := range w {
			vals = append(vals, c)
		}
		ceiling := cluster.Median(vals)
		return decide(tally(resps, func(r cluster.Response) float64 { return math.Min(w[r.Node], ceiling) }))
	}}
}

// MaxConfidence takes the single most confident answer. It is here as the
// cautionary baseline: it hands the decision to whoever claims the most.
func MaxConfidence() cluster.Aggregator {
	return stateless{"max-confidence", func(_ cluster.Request, resps []cluster.Response) cluster.Decision {
		w := confWeights(resps)
		buckets, _ := tally(resps, one)
		if len(buckets) == 0 {
			return cluster.Decision{Abstained: true}
		}
		best := func(b *bucket) (m float64) {
			for _, n := range b.nodes {
				m = math.Max(m, w[n])
			}
			return
		}
		sort.SliceStable(buckets, func(i, j int) bool { return best(buckets[i]) > best(buckets[j])+eps })
		d := cluster.Decision{Tokens: buckets[0].tokens, Winners: buckets[0].nodes}
		d.Share = float64(len(buckets[0].nodes)) / float64(len(cluster.Answered(resps)))
		if len(buckets) > 1 {
			d.Margin = best(buckets[0]) - best(buckets[1])
		} else {
			d.Margin = 1
		}
		return d
	}}
}
