package engine

import (
	"fmt"

	"kllm/engine/backend"
)

// GenEvent is one streamed generation event.
type GenEvent struct {
	Token int32
	Err   error
	Done  bool
}

type reqState struct {
	prompt    []int32
	maxNew    int
	out       chan GenEvent
	seq       *Sequence
	prefilled bool
	next      int32 // pending token to feed (after prefill)
	generated int
}

// Scheduler implements continuous (in-flight) batching: one goroutine owns
// the GPU, forming a fresh batch every step from whatever sequences are
// running, admitting queued requests as batch slots and KV blocks free up.
type Scheduler struct {
	e        *Engine
	maxBatch int
	queue    chan *reqState
	stopCh   chan struct{}
	doneCh   chan struct{}
}

func NewScheduler(e *Engine, maxBatch int) *Scheduler {
	if maxBatch <= 0 || maxBatch > backend.MaxBatchSeqs {
		maxBatch = 8
	}
	return &Scheduler{
		e:        e,
		maxBatch: maxBatch,
		queue:    make(chan *reqState, 1024),
		stopCh:   make(chan struct{}),
		doneCh:   make(chan struct{}),
	}
}

// Submit queues a generation request; the returned channel streams one event
// per token and closes after Done (buffered: the scheduler never blocks on a
// slow consumer).
func (s *Scheduler) Submit(prompt []int32, maxNew int) (<-chan GenEvent, error) {
	if len(prompt) == 0 || maxNew <= 0 {
		return nil, fmt.Errorf("scheduler: empty prompt or non-positive maxNew")
	}
	if len(prompt) > s.e.MaxStepTokens() {
		return nil, fmt.Errorf("scheduler: prompt of %d tokens exceeds per-step budget %d",
			len(prompt), s.e.MaxStepTokens())
	}
	r := &reqState{
		prompt: prompt,
		maxNew: maxNew,
		out:    make(chan GenEvent, maxNew+1),
	}
	select {
	case s.queue <- r:
		return r.out, nil
	case <-s.stopCh:
		return nil, fmt.Errorf("scheduler stopped")
	}
}

func (s *Scheduler) Start() { go s.run() }

// Stop halts the loop after the in-flight step; running requests get an
// error event.
func (s *Scheduler) Stop() {
	close(s.stopCh)
	<-s.doneCh
}

func (s *Scheduler) run() {
	defer close(s.doneCh)
	var running, waiting []*reqState

	fail := func(r *reqState, err error) {
		if r.seq != nil {
			r.seq.Release()
		}
		r.out <- GenEvent{Err: err, Done: true}
		close(r.out)
	}
	finish := func(r *reqState) {
		r.seq.Release()
		r.out <- GenEvent{Done: true}
		close(r.out)
	}

	for {
		// Idle: block until work arrives or we are stopped.
		if len(running) == 0 && len(waiting) == 0 {
			select {
			case r := <-s.queue:
				waiting = append(waiting, r)
			case <-s.stopCh:
				return
			}
		}
		// Drain whatever else is queued right now (non-blocking).
	drain:
		for {
			select {
			case r := <-s.queue:
				waiting = append(waiting, r)
			case <-s.stopCh:
				for _, r := range append(running, waiting...) {
					fail(r, fmt.Errorf("scheduler stopped"))
				}
				return
			default:
				break drain
			}
		}

		// Admit waiting requests into free batch slots.
		for len(running) < s.maxBatch && len(waiting) > 0 {
			r := waiting[0]
			if r.seq == nil {
				r.seq = s.e.NewSequence()
			}
			running = append(running, r)
			waiting = waiting[1:]
		}

		// Form this step's batch: prompts for new sequences, one token for
		// the rest, bounded by the backend's per-step token budget. New
		// sequences that can't fit (tokens or KV blocks) wait for the next
		// step; established sequences failing Reserve are failed outright
		// (eviction/preemption is future work).
		budget := s.e.MaxStepTokens()
		batch := make([]backend.SeqForward, 0, len(running))
		stepReqs := make([]*reqState, 0, len(running))
		var skipped []*reqState // in-flight but not in this step's batch
		for _, r := range running {
			tokens := []int32{r.next}
			if !r.prefilled {
				tokens = r.prompt
			}
			if len(tokens) > budget {
				if !r.prefilled {
					waiting = append(waiting, r) // no room this step; re-admit later
				} else {
					skipped = append(skipped, r) // decodes retry next step
				}
				continue
			}
			sf, err := r.seq.Step(tokens)
			if err != nil {
				if !r.prefilled {
					waiting = append(waiting, r) // retry when blocks free up
				} else {
					fail(r, fmt.Errorf("kv reservation failed mid-stream: %w", err))
				}
				continue
			}
			budget -= len(tokens)
			batch = append(batch, sf)
			stepReqs = append(stepReqs, r)
		}
		if len(batch) == 0 {
			running = skipped
			continue
		}

		logits, err := s.e.B.ForwardBatch(batch)
		if err != nil {
			for _, r := range append(stepReqs, skipped...) {
				fail(r, err)
			}
			running = nil
			continue
		}

		next := skipped
		for i, r := range stepReqs {
			r.seq.Commit(len(batch[i].Tokens))
			r.prefilled = true
			tok := Argmax(logits[i])
			r.next = tok
			r.generated++
			r.out <- GenEvent{Token: tok}
			eos := s.e.Cfg.EOSTokenID >= 0 && int64(tok) == s.e.Cfg.EOSTokenID
			if eos || r.generated >= r.maxNew {
				finish(r)
				continue
			}
			next = append(next, r)
		}
		running = next
	}
}
