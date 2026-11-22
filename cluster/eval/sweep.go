package eval

import (
	"context"
	"fmt"
	"runtime"
	"sync"

	"kllm/cluster/dataset"
)

// Sweep is the standard experiment: every attack at every adversary count,
// scored under every method.
type Sweep struct {
	Nodes          int      `json:"nodes"`
	MaxAdversaries int      `json:"max_adversaries"` // adversary counts run from 1 to this
	Requests       int      `json:"requests"`        // per condition, per seed
	Seeds          []uint64 `json:"seeds"`           // each seed is an independent model, workload and cluster; results pool
	Attacks        []string `json:"attacks"`
	Methods        []Method `json:"methods"`
}

// WithDefaults fills unset fields: 7 nodes, adversaries up to a bare
// majority (so the tables show where each method breaks, not just that it
// holds), 1000 requests, 5 seeds, every attack, every method.
func (s Sweep) WithDefaults() Sweep {
	if s.Nodes <= 0 {
		s.Nodes = 7
	}
	if s.MaxAdversaries <= 0 {
		s.MaxAdversaries = s.Nodes/2 + 1
	}
	s.MaxAdversaries = min(s.MaxAdversaries, s.Nodes-1)
	if s.Requests <= 0 {
		s.Requests = 1000
	}
	if len(s.Seeds) == 0 {
		s.Seeds = []uint64{1, 2, 3, 4, 5}
	}
	if len(s.Attacks) == 0 {
		s.Attacks = dataset.Attacks
	}
	if len(s.Methods) == 0 {
		s.Methods = DefaultMethods(s.Nodes)
	}
	return s
}

// Report is a sweep's results.
type Report struct {
	Sweep     Sweep       `json:"sweep"`
	Metrics   []Metrics   `json:"metrics"`   // one per condition x method
	Detection []Detection `json:"detection"` // one per condition
}

type condition struct {
	attack string
	f      int
}

// Run generates every condition's datasets and scores every method on
// them. Conditions run in parallel; the result does not depend on it.
// progress, if set, is called as each condition finishes.
func (s Sweep) Run(ctx context.Context, progress func(done, total int, name string)) (Report, error) {
	s = s.WithDefaults()
	if err := Validate(s.Methods); err != nil {
		return Report{}, err
	}
	conds := []condition{{"normal", 0}}
	for _, a := range s.Attacks {
		for f := 1; f <= s.MaxAdversaries; f++ {
			conds = append(conds, condition{a, f})
		}
	}

	type result struct {
		metrics []Metrics
		det     Detection
		err     error
	}
	results := make([]result, len(conds))
	jobs := make(chan int)
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for w := 0; w < max(1, runtime.GOMAXPROCS(0)); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i].metrics, results[i].det, results[i].err = s.run(ctx, conds[i])
				mu.Lock()
				done++
				if progress != nil {
					progress(done, len(conds), fmt.Sprintf("%s-f%d", conds[i].attack, conds[i].f))
				}
				mu.Unlock()
			}
		}()
	}
	for i := range conds {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	rep := Report{Sweep: s}
	for _, r := range results {
		if r.err != nil {
			return Report{}, r.err
		}
		rep.Metrics = append(rep.Metrics, r.metrics...)
		rep.Detection = append(rep.Detection, r.det)
	}
	return rep, nil
}

func (s Sweep) run(ctx context.Context, c condition) ([]Metrics, Detection, error) {
	scores := make([]score, len(s.Methods))
	var det detScore
	for _, seed := range s.Seeds {
		sc, err := dataset.Standard(c.attack, s.Nodes, c.f, s.Requests, seed)
		if err != nil {
			return nil, Detection{}, err
		}
		recs, _, err := dataset.Collect(ctx, sc)
		if err != nil {
			return nil, Detection{}, err
		}
		for i, m := range s.Methods {
			if err := scores[i].replay(recs, m); err != nil { // fresh strategy state per seed
				return nil, Detection{}, err
			}
		}
		det.replay(recs)
	}
	out := make([]Metrics, len(s.Methods))
	for i, m := range s.Methods {
		out[i] = scores[i].metrics(m)
		out[i].Attack, out[i].Adversaries, out[i].Nodes = c.attack, c.f, s.Nodes
	}
	d := det.detection()
	d.Attack, d.Adversaries, d.Nodes = c.attack, c.f, s.Nodes
	return out, d, nil
}

// Get returns one cell of the sweep.
func (r Report) Get(attack string, adversaries int, method string) (Metrics, bool) {
	for _, m := range r.Metrics {
		if m.Attack == attack && m.Adversaries == adversaries && m.Method == method {
			return m, true
		}
	}
	return Metrics{}, false
}

// Robustness summarises one method across the whole sweep.
type Robustness struct {
	Method string
	Clean  float64   // exact match with no adversary
	Worst  []float64 // Worst[f-1]: lowest exact match over all attacks with f adversaries
	Hard   []string  // Hard[f-1]: the attack that achieved it

	// WorstSeed[f-1] is the lowest exact match of any single seed in that
	// cell. Where it sits far below Worst, the pooled figure is an average
	// of runs that held and runs that were captured, and which attack
	// shows the lowest one is largely the luck of the seeds.
	WorstSeed []float64

	// Breakdown is the smallest adversary count at which some attack
	// pushes exact match below one half; 0 if none in the sweep does.
	Breakdown int
}

// Robustness computes the summary for every wait-for-all method.
func (r Report) Robustness() []Robustness {
	var out []Robustness
	for _, m := range r.Sweep.Methods {
		if m.PolicyName() != "all" || isAblation(m.Aggregator) {
			continue
		}
		rb := Robustness{Method: m.String()}
		if c, ok := r.Get("normal", 0, rb.Method); ok {
			rb.Clean = c.ExactMatch
		}
		for f := 1; f <= r.Sweep.MaxAdversaries; f++ {
			worst, seed, hard := 2.0, 0.0, ""
			for _, a := range r.Sweep.Attacks {
				if c, ok := r.Get(a, f, rb.Method); ok && c.ExactMatch < worst {
					worst, seed, hard = c.ExactMatch, c.SeedMin, a
				}
			}
			rb.Worst, rb.Hard, rb.WorstSeed = append(rb.Worst, worst), append(rb.Hard, hard), append(rb.WorstSeed, seed)
			if worst < 0.5 && rb.Breakdown == 0 {
				rb.Breakdown = f
			}
		}
		out = append(out, rb)
	}
	return out
}
