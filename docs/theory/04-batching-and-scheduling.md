# 4 · Batching and scheduling

## 4.1 Why batching is almost-free throughput

Decode at batch 1 reads every weight matrix to do one vector's worth of
math (§2.2). Run $B$ sequences *in the same forward pass* and each weight
matrix is read **once** but used $B$ times: the batched GEMM computes
$Y_{B\times \text{out}} = X_{B\times \text{in}} W^\top$, and the weight
traffic — the thing that bounds decode — is unchanged. To first order:

$$\text{step time}(B) \;\approx\; \underbrace{\tau_{\text{mem}}}_{\text{weights+KV reads}} + \underbrace{B\cdot \tau_{\text{compute}}}_{\text{tiny until large } B} \qquad\Rightarrow\qquad \text{throughput}(B) \approx \frac{B}{\tau_{\text{mem}}}\ \text{for small } B$$

Throughput scales *linearly* while latency stays *flat*, until one of three
walls: (i) arithmetic intensity rises enough to hit the compute roof
(Ch. 5) — for weights this takes $B$ in the hundreds; (ii) **KV reads**,
which are per-sequence and do *not* amortize, start to dominate
($B \times$ cache traffic grows linearly); (iii) memory capacity for $B$
caches runs out (Ch. 3). In the launch-overhead-bound regime of small
models, batching is even freer: the fixed per-step launch cost is divided
by $B$ tokens. kllm's measured curves show the textbook shape twice —
1.1K → 23K tok/s (tiny model, GTX 1650) and 81 → 1804 tok/s
(TinyLlama-1.1B, A100), both flattening at the configured batch cap.

The economic statement: **single-stream tok/s is a latency spec; system
tok/s is a throughput spec; batching buys the second at almost no cost to
the first** — until queueing (§4.4) says otherwise.

## 4.2 Static batching and its two pathologies

The naive scheme — collect $B$ requests, run the batch to completion —
fails on the *variance* of real workloads:

1. **Convoy at the tail:** sequences finish at different times (different
   generation lengths); finished slots idle until the *longest* member
   completes. With completion lengths $\ell_i$, utilization is
   $\sum \ell_i / (B \max_i \ell_i)$ — a heavy-tailed length distribution
   makes this arbitrarily bad.
2. **Admission latency:** a request arriving just after a batch forms
   waits an entire batch lifetime before its prefill even starts.

Both pathologies come from batching at *request* granularity.

## 4.3 Continuous batching: iterate at token granularity

The fix (Orca, Yu et al. 2022 — "iteration-level scheduling"): re-form the
batch **every decode step**. The batch is a churning set, not a cohort:

- a sequence that emits EOS or hits its token limit leaves *this step*,
  freeing its slot and its KV blocks immediately;
- a queued request joins at the next step boundary — its prefill (a
  many-token span) rides in the same `forward_step` as everyone else's
  single-token decodes (§2.4 made prefill and decode the same operation
  precisely so this composes);
- the GPU-facing loop is one thread of control: form batch → one
  `forward_step` → scatter tokens to streams → repeat.

This turns the two pathologies into non-events and makes the *scheduler*
the interesting component. Design invariants kllm's implementation commits
to, each of which is a general lesson:

- **One owner for the device.** A single scheduler goroutine issues every
  forward call; concurrency lives in the request/stream layer (channels),
  never in the GPU layer. Eliminates a whole class of interleaving bugs and
  matches the hardware reality that steps serialize anyway.
- **Two admission budgets, not one.** A step is constrained by *batch
  slots* (max sequences the kernels handle) **and** *step tokens* (scratch
  and prefill compute). A slots-only scheduler admits 32 prefills of 48
  tokens into a 256-token scratch and dies — kllm found exactly this under
  load. vLLM's `max_num_batched_tokens` is the same budget. Decodes cost 1
  token each, so bounding tokens also bounds prefill's latency intrusion on
  running streams ("chunked prefill" in the literature splits long prompts
  across steps for the same reason).
- **Backpressure, not blocking.** Streams get buffered channels sized to
  their token limit; a slow consumer can never stall the batch loop.
- **Failure containment.** A sequence that can't get KV blocks mid-stream
  fails *alone*; prefills that don't fit simply wait. (The upgrade path is
  preemption — evict a victim's blocks and re-prefill later — which trades
  recompute for capacity; kllm documents rather than implements it.)

## 4.4 Latency under load: the queueing view

Serving metrics are distributional, and the right basic lenses are
queueing-theoretic:

- **TTFT** = queue wait + prefill time. Under load it is dominated by the
  *wait*, and grows superlinearly as utilization $\rho \to 1$ (in an
  M/M/1-flavored approximation, $W \propto \rho/(1-\rho)$) — kllm's sweep
  shows TTFT p50 going 2.4 → 5.1 → 18 ms as offered concurrency passes the
  batch cap, the classic knee.
- **ITL** = step time, inflated by batch size only mildly (§4.1) — until
  a co-scheduled *prefill* lands in your step and stretches it (the
  token-budget knob bounds this).
- **Little's law** ($\bar{N} = \lambda \bar{W}$) ties them: at arrival rate
  $\lambda$ with mean residence $\bar W$, the mean in-system count is their
  product — if $\bar N$ exceeds slots + queue tolerance, you shed load or
  scale out. It also gives the capacity planner's identity: system tok/s
  $\approx$ (slots) × (1/ITL) at saturation.
- **Goodput vs throughput:** past the saturation knee, admitted load only
  moves wait time, not tok/s — kllm's curve is flat from concurrency 32 to
  64 while p99 doubles. The optimal operating point for latency-SLO serving
  is *at* the knee, not past it.

## 4.5 The Go-shaped part

Continuous batching is a concurrency-orchestration problem more than a GPU
problem, which is the argument for a systems language on top of the CUDA
core: requests are goroutines, the queue is a channel, streams are
channels, the scheduler is a select loop, and the entire policy surface
(admission order, budgets, retirement, metrics) is ordinary testable code.
The oracle-gated test that matters: N concurrent streams through the
scheduler must each reproduce their solo (and reference) token sequence
exactly — batching must be **observationally invisible** except in timing.

## Where this lives in kllm

- `engine/scheduler.go` — the loop of §4.3 verbatim: single goroutine,
  channel queue, dual budgets, per-step admission/retirement, buffered
  streams, per-request failure.
- `server/` + `cmd/loadgen` — SSE streaming and the measurement harness
  that produced the throughput/latency curves; `engine/metrics` exports
  TTFT/ITL histograms, queue depth, KV utilization (Prometheus + JSON).
- Journal entries: the token-budget bug found at concurrency 32; both
  saturation curves; `TestSchedulerMatchesHF`.

## Reading list

1. Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022 — iteration-level scheduling; the founding paper of continuous batching.
2. Kwon et al., vLLM (2023) §4 — scheduling with paged memory; `max_num_batched_tokens`.
3. Agrawal et al., *Sarathi-Serve: Taming Throughput-Latency Tradeoff*, 2024 — chunked prefill and stall-free scheduling, the refinement of the token budget.
4. Harchol-Balter, *Performance Modeling and Design of Computer Systems* — the queueing theory behind §4.4, properly.
