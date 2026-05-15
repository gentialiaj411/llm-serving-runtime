# ADR-0001: Scheduler Objective

- Status: Proposed
- Date: 2026-05-15

## Context
Need explicit optimization objective before continuous batching complexity increases.

## Decision
Use throughput-maximizing objective constrained by deadline classes and p99 latency SLO.

## Consequences
Enables consistent admission/scheduling tradeoffs and benchmark comparability.
