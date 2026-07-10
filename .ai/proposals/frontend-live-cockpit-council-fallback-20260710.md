# Council fallback proposal: frontend live cockpit

Kimi CLI was invoked in read-only plan mode for an independent review but
returned HTTP 402 during membership verification. This record is an explicit
fallback and does not claim a completed Kimi review.

## 1. Strongest part of the Codex proposal

The readback-first rule correctly treats data provenance and write authority as
product behavior, not as secondary status text.

## 2. Possible blind spots

- Audio-capture event state must win over late dashboard snapshots.
- Mobile mode gates must match desktop gates.
- An offline demo may simulate modes only while it is unmistakably labeled as
  mock data.
- Visual polish cannot compensate for unverified physical console writes.

## 3. Complexity, fragility or cost

Adding a second frontend state store for telemetry would create drift. The
smallest robust approach is to merge canonical WebSocket events into the
existing Product Layer payload model and mark the newest capture status as
authoritative.

## 4. What should be kept

Keep the existing console layout, existing tokens, operator proposal workflow,
observation-only default and fail-closed backend mode policy.

## 5. Revised recommendation

Accept the implementation if desktop/mobile mode gating is identical, all
active buttons reach backend contracts, stale capture races are covered, live
missing data never falls back to mock numbers, and the final score reserves
points for bundle splitting and unproven broad write control.
