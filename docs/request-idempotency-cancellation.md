# Request Idempotency and Cancellation Semantics

This document defines request-level behavior for `runtime/phase2/coordinator_server.py`.

## Request ID semantics
- `request_id` is the idempotency key.
- If a request with the same `request_id` has already completed and is present in the coordinator completion cache, the coordinator returns that cached terminal response.
- Cached replay applies before worker routing.

## Cancellation semantics
- `POST /v1/requests/{request_id}/cancel` always returns:
- `{"request_id":"<id>","status":"cancel_accepted"}`
- Cancellation is idempotent at the API level: repeated cancel calls keep returning `cancel_accepted`.
- If the request is currently active on a known worker, the coordinator attempts best-effort forwarding to `POST /cancel/{request_id}` on that worker.
- If the request is not active, cancellation is still recorded in the coordinator cancellation cache.

## Interaction between cancellation and replay
- If a `request_id` is marked cancelled and there is no cached terminal completion for that request, subsequent `/v1/chat/completions` calls with that same `request_id` are rejected with HTTP `499`.
- If a cached terminal completion exists for that `request_id`, replay returns the cached completion.

## Retry boundary
- Streaming retries occur only before first token emission.
- After first emitted token, coordinator stream retry is intentionally conservative and will not mask all failures.
