# Orcaforge Design Notes

This document is the short systems-interview version of the runtime choices. It focuses on what the prototype is trying to prove, where the real serving constraints show up, and which recovery boundaries are deliberate.

## Iteration-Level Scheduling

Orcaforge follows the Orca paper's core scheduling idea: admit work at decode iteration boundaries instead of waiting for an entire batch to drain. In autoregressive inference, a request alternates between a prefill phase and a long sequence of single-token decode steps. If the scheduler treats the whole generation as an indivisible batch job, short requests arriving behind long generations wait unnecessarily and GPU slots sit underutilized as sequences finish at different lengths.

Iteration-level scheduling makes the decode loop the scheduling boundary. At each token step, completed/cancelled/timed-out requests can leave, and newly admitted requests can join the next iteration. This gives the runtime a practical lever for:

- reducing head-of-line blocking from long generations
- improving occupancy when sequence lengths diverge
- enforcing cancellation and deadlines without waiting for a batch to finish
- making streaming token emission natural, because each decode step has a visible output boundary

The Python worker models this as a one-token-per-active-request loop. It is not a production kernel implementation, but it preserves the scheduling contract the C++/CUDA path should keep.

## Paged KV Allocation

Contiguous KV allocation is simple but brittle under real traffic. Requests have dynamic prompt lengths, dynamic generation lengths, and unpredictable cancellation/deadline behavior. If each sequence needs one contiguous KV region sized for its worst case, the allocator either over-reserves memory or accumulates holes that are hard to reuse.

Paged KV allocation breaks a sequence's cache into fixed-size blocks. A request owns a logical list of blocks, while physical blocks can come from a free list. That buys:

- fragmentation resistance as requests of different lengths finish out of order
- incremental growth as decode length increases
- cheap release on completion, cancellation, or timeout
- allocator metrics that map to serving pressure: used blocks, free blocks, occupancy, and fragmentation

The current allocator is intentionally small, but the interface matches the production direction: admit only when the required blocks are available, release blocks on every terminal path, and expose allocator state for the coordinator.

## Streaming Retry Boundary

Coordinator retry is safe only before the first assistant token reaches the client. Before that point, the request is still externally invisible: if the selected worker dies or rejects the request, the coordinator can select another healthy worker and replay the request without corrupting the client-visible stream.

After the first token is emitted, retry becomes unsafe. The client has already observed a prefix. A different worker may produce a different continuation, repeat tokens, or diverge because of kernel nondeterminism, batching differences, or model state. There is no general way to merge those partial outputs into one correct OpenAI-compatible stream.

For that reason, Orcaforge's streaming path retries transport failures before first token emission and intentionally fails the stream after partial output has been sent. This is the conservative recovery boundary: preserve correctness over hiding every failure.

## Coordinator Routing

The coordinator keeps a health view of workers through background `/healthz` probes. Admission uses only currently healthy workers and selects the least-inflight worker among them. This is deliberately simple and robust:

- health checks remove dead or wedged workers from normal routing
- least-inflight selection avoids piling new work onto the busiest healthy worker
- request IDs let duplicate non-streaming requests reuse completed responses when cached
- cancellation routes to the worker currently recorded for that active request

The policy is process-local and meant for a single coordinator prototype. A production deployment would externalize health, logs, and replay state, but the local behavior demonstrates the control-plane contract: route by health, balance by current load, and avoid unsafe replay after client-visible streaming output.
