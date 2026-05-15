# ADR-0003: Transport Contract

- Status: Proposed
- Date: 2026-05-15

## Context
Hot path cannot rely on heavy reflection or unstable envelopes.

## Decision
Use versioned gRPC messages with compact payloads (token IDs, sampling params, deadlines, cancellation IDs) and explicit compatibility policy.

## Consequences
Reduces future breakage and keeps coordinator/worker decoupled.
