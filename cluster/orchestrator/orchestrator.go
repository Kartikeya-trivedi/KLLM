// Package orchestrator coordinates a cluster of inference nodes: it keeps
// the membership and its health, fans each request out under a dispatch
// policy, hands the collected answers to an aggregator, and emits one
// Result per request for logging.
package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"kllm/cluster"
	"kllm/cluster/node"
	"kllm/engine/metrics"
)

// ErrNoNodes is returned when no healthy node is available.
var ErrNoNodes = errors.New("orchestrator: no healthy nodes")

// Truster is implemented by aggregators that learn how far to trust each
// node. The orchestrator uses it for trust-ordered dispatch and reports it
// on /nodes.
type Truster interface {
	Trust(node string) float64
}

// Options configures an Orchestrator.
type Options struct {
	Policy cluster.Policy

	// Aggregator combines the answers. Nil means collect only: the Result
	// carries every response and no Decision — the mode datasets are
	// generated in.
	Aggregator cluster.Aggregator

	// Sink receives every Result (the request log). Called synchronously,
	// in completion order.
	Sink func(Result)

	// HealthEvery is the probe interval for Run; FailAfter consecutive
	// probe failures take a node out of rotation until it answers again.
	HealthEvery time.Duration
	FailAfter   int
}

// Result is everything that happened to one request.
type Result struct {
	Request    cluster.Request    `json:"request"`
	Responses  []cluster.Response `json:"responses"`
	Decision   *cluster.Decision  `json:"decision,omitempty"`
	Aggregator string             `json:"aggregator,omitempty"`
	Policy     string             `json:"policy"`
	LatencyMs  float64            `json:"latency_ms"`
	Queried    int                `json:"queried"`
	Rounds     int                `json:"rounds"`
}

// MemberInfo describes one node as the orchestrator sees it.
type MemberInfo struct {
	ID        string  `json:"id"`
	URL       string  `json:"url,omitempty"`
	Healthy   bool    `json:"healthy"`
	Requests  uint64  `json:"requests"`
	Errors    uint64  `json:"errors"`
	LatencyMs float64 `json:"latency_ewma_ms"`
	Trust     float64 `json:"trust"`
}

type member struct {
	client   node.Client
	healthy  atomic.Bool
	fails    atomic.Int32
	requests metrics.Counter
	errs     metrics.Counter
	latency  metrics.Gauge // EWMA, ms
}

func (m *member) observe(r cluster.Response) {
	m.requests.Inc()
	if !r.OK() {
		m.errs.Inc()
		return
	}
	if prev := m.latency.Value(); prev == 0 {
		m.latency.Set(r.LatencyMs)
	} else {
		m.latency.Set(0.2*r.LatencyMs + 0.8*prev)
	}
}

// Orchestrator is safe for concurrent use.
type Orchestrator struct {
	opts Options

	mu      sync.RWMutex
	members map[string]*member

	seq       atomic.Uint64
	requests  metrics.Counter
	failed    metrics.Counter
	abstained metrics.Counter
	nodeCalls metrics.Counter
	latency   *metrics.Histogram // seconds
	sinkMu    sync.Mutex
}

func New(opts Options) *Orchestrator {
	if opts.HealthEvery <= 0 {
		opts.HealthEvery = 2 * time.Second
	}
	if opts.FailAfter <= 0 {
		opts.FailAfter = 2
	}
	return &Orchestrator{
		opts:    opts,
		members: map[string]*member{},
		latency: metrics.NewHistogram([]float64{.01, .025, .05, .075, .1, .15, .2, .3, .5, .75, 1, 2, 5, 10}),
	}
}

// Add registers a node (or replaces the one with the same id). New nodes
// start healthy; the prober corrects that within FailAfter intervals.
func (o *Orchestrator) Add(c node.Client) {
	m := &member{client: c}
	m.healthy.Store(true)
	o.mu.Lock()
	o.members[c.ID()] = m
	o.mu.Unlock()
}

// Remove drops a node from the membership.
func (o *Orchestrator) Remove(id string) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	_, ok := o.members[id]
	delete(o.members, id)
	return ok
}

func (o *Orchestrator) trust(id string) float64 {
	if t, ok := o.opts.Aggregator.(Truster); ok {
		return t.Trust(id)
	}
	return 1
}

// Members lists the membership, sorted by id.
func (o *Orchestrator) Members() []MemberInfo {
	o.mu.RLock()
	defer o.mu.RUnlock()
	out := make([]MemberInfo, 0, len(o.members))
	for id, m := range o.members {
		info := MemberInfo{
			ID: id, Healthy: m.healthy.Load(), Trust: o.trust(id),
			Requests: m.requests.Value(), Errors: m.errs.Value(), LatencyMs: m.latency.Value(),
		}
		if u, ok := m.client.(interface{ URL() string }); ok {
			info.URL = u.URL()
		}
		out = append(out, info)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func (o *Orchestrator) healthy() map[string]*member {
	o.mu.RLock()
	defer o.mu.RUnlock()
	out := make(map[string]*member, len(o.members))
	for id, m := range o.members {
		if m.healthy.Load() {
			out[id] = m
		}
	}
	return out
}

// Run probes every node until ctx is done, taking unresponsive ones out of
// rotation and putting them back when they recover.
func (o *Orchestrator) Run(ctx context.Context) {
	t := time.NewTicker(o.opts.HealthEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			o.Probe(ctx)
		}
	}
}

// Probe checks every node once.
func (o *Orchestrator) Probe(ctx context.Context) {
	o.mu.RLock()
	all := make([]*member, 0, len(o.members))
	for _, m := range o.members {
		all = append(all, m)
	}
	o.mu.RUnlock()

	var wg sync.WaitGroup
	for _, m := range all {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pctx, cancel := context.WithTimeout(ctx, o.opts.HealthEvery)
			defer cancel()
			if _, err := m.client.Stats(pctx); err != nil {
				if int(m.fails.Add(1)) >= o.opts.FailAfter {
					m.healthy.Store(false)
				}
				return
			}
			m.fails.Store(0)
			m.healthy.Store(true)
		}()
	}
	wg.Wait()
}

// Stats samples every node's self-reported stats (for the system-metrics
// log). Nodes that fail to answer are omitted.
func (o *Orchestrator) Stats(ctx context.Context) []node.Stats {
	o.mu.RLock()
	clients := make([]node.Client, 0, len(o.members))
	for _, m := range o.members {
		clients = append(clients, m.client)
	}
	o.mu.RUnlock()

	out := make([]node.Stats, len(clients))
	ok := make([]bool, len(clients))
	var wg sync.WaitGroup
	for i, c := range clients {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, err := c.Stats(ctx)
			out[i], ok[i] = s, err == nil
		}()
	}
	wg.Wait()
	kept := out[:0]
	for i, s := range out {
		if ok[i] {
			kept = append(kept, s)
		}
	}
	sort.Slice(kept, func(i, j int) bool { return kept[i].Node < kept[j].Node })
	return kept
}

// Infer runs one request through the cluster under the configured policy.
func (o *Orchestrator) Infer(ctx context.Context, req cluster.Request) (Result, error) {
	if len(req.Prompt) == 0 {
		return Result{}, errors.New("orchestrator: empty prompt")
	}
	if req.MaxNewTokens <= 0 {
		req.MaxNewTokens = 64
	}
	if req.ID == "" {
		req.ID = fmt.Sprintf("req-%d", o.seq.Add(1))
	}
	o.requests.Inc()

	members := o.healthy()
	if len(members) == 0 {
		o.failed.Inc()
		return Result{Request: req}, ErrNoNodes
	}
	ids := make([]string, 0, len(members))
	for id := range members {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	p := o.opts.Policy.Normalize(len(ids))
	order := cluster.Order(p, req, ids, o.trust)
	pick := func(ids []string) []*member {
		ms := make([]*member, len(ids))
		for i, id := range ids {
			ms[i] = members[id]
		}
		return ms
	}

	res := Result{Request: req, Policy: p.String()}
	start := time.Now()
	switch p.Mode {
	case cluster.ModeFirstK:
		res.Responses = o.ask(ctx, req, pick(order), p, p.K)
		res.Queried, res.Rounds = len(order), 1

	case cluster.ModeAdaptive:
		for asked := 0; asked < len(order); {
			n := p.Step
			if asked == 0 {
				n = p.Initial
			}
			n = min(n, len(order)-asked)
			round := o.ask(ctx, req, pick(order[asked:asked+n]), p, n)
			asked += n
			res.Responses = append(res.Responses, round...)
			res.Queried, res.Rounds = asked, res.Rounds+1
			if _, margin, answered := cluster.Agreement(res.Responses); answered >= p.Initial && margin >= p.Margin {
				break
			}
		}

	default:
		res.Responses = o.ask(ctx, req, pick(order), p, len(order))
		res.Queried, res.Rounds = len(order), 1
	}
	res.LatencyMs = float64(time.Since(start)) / float64(time.Millisecond)
	if p.Mode == cluster.ModeAll {
		// Virtual-time nodes answer instantly and report their latency;
		// under wait-for-all the request took as long as the slowest.
		for _, r := range res.Responses {
			res.LatencyMs = max(res.LatencyMs, r.LatencyMs)
		}
	}
	o.nodeCalls.Add(uint64(res.Queried))
	o.latency.Observe(res.LatencyMs / 1000)

	if agg := o.opts.Aggregator; agg != nil {
		received := make([]cluster.Response, 0, len(res.Responses))
		for _, r := range res.Responses {
			if r.Err != cluster.ErrCancelled { // being cancelled says nothing about a node
				received = append(received, r)
			}
		}
		d := agg.Aggregate(req, cluster.Blind(received))
		res.Decision, res.Aggregator = &d, agg.Name()
		if d.Abstained {
			o.abstained.Inc()
		}
	}
	if o.opts.Sink != nil {
		o.sinkMu.Lock()
		o.opts.Sink(res)
		o.sinkMu.Unlock()
	}
	return res, nil
}

// ask sends req to nodes concurrently and returns once `need` of them have
// answered or all have finished, cancelling whoever is still working. The
// result has one entry per node, in the order given.
func (o *Orchestrator) ask(ctx context.Context, req cluster.Request, nodes []*member, p cluster.Policy, need int) []cluster.Response {
	timeout := time.Duration(p.TimeoutMs * float64(time.Millisecond))
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	type arrival struct {
		i int
		r cluster.Response
	}
	ch := make(chan arrival, len(nodes)) // buffered: late arrivals never block
	start := time.Now()
	for i, m := range nodes {
		go func() { ch <- arrival{i, m.client.Infer(cctx, req)} }()
	}

	out := make([]cluster.Response, len(nodes))
	got := make([]bool, len(nodes))
	for done, ok := 0, 0; done < len(nodes) && ok < need; done++ {
		a := <-ch
		r := a.r
		if r.OK() && r.LatencyMs > p.TimeoutMs { // a virtual-time node past the deadline
			r.Err, r.Tokens, r.Conf, r.LatencyMs = context.DeadlineExceeded.Error(), nil, nil, p.TimeoutMs
		}
		nodes[a.i].observe(r)
		out[a.i], got[a.i] = r, true
		if r.OK() {
			ok++
		}
	}
	cancel()
	elapsed := float64(time.Since(start)) / float64(time.Millisecond)
	for i := range out {
		if !got[i] {
			out[i] = cluster.Response{RequestID: req.ID, Node: nodes[i].client.ID(), Err: cluster.ErrCancelled, LatencyMs: elapsed}
		}
	}
	return out
}
