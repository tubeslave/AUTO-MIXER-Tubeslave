# Correction Methods Backlog

Purpose: keep the full mixer-correction goal visible while preventing broad
method work from bypassing the supervised WING MVP.

Current rule: do not live-apply any family below except the existing supervised
fader/gain path after the MVP proof gate. Before then, build methods as
offline, dry-run, proposal-only, read-only, and tested artifacts.

| Method family | Current stage | Next useful work | Live gate |
| --- | --- | --- | --- |
| Gain / Trim | supervised_candidate_partial | Finish current WING read-only preflight, then supervised gain-down proof with physical readback and rollback. | `Docs/MVP_WING_SUPERVISED.md` |
| Fader / Balance | supervised_candidate_partial | Finish current WING read-only preflight, then supervised fader proof with physical readback and rollback. | `Docs/MVP_WING_SUPERVISED.md` |
| Pan / Stereo Placement | proposal_only | Keep spatial pan proposals non-queueable with fresh current-pan context, mono/phase safety for non-center moves, source-identity trust, and rollback/readback requirements. | separate ADR after WING MVP |
| EQ | proposal_only | Keep corrective EQ proposals non-queueable with fresh EQ-band identity, feedback-notch/manual-band protection, bounded gain/headroom, fresh phase context, source-identity trust, and rollback/readback requirements. | separate ADR after WING MVP |
| Cross-channel unmasking | proposal_only | Keep linked mirror-EQ proposals arrangement-priority aware, masking/source-identity freshness-gated, non-queueable, and blocked from live apply until EQ-band readback, sum-bus, rollback, and hardware proof are complete. | separate ADR after WING MVP |
| Compressor | proposal_only | Keep dynamics proposals non-queueable with GR estimate, true-peak bounded makeup gain, feedback safety, fresh dynamics context, source-identity trust, and rollback/readback requirements. | separate ADR after WING MVP |
| De-essing | proposal_only | Keep vocal sibilance proposals arrangement-aware, freshness-gated, and non-queueable until dynamics/dynamic-EQ readback and rollback are proven. | separate ADR after WING MVP |
| Gate / Expander | proposal_only | Keep context-aware gate/expander proposals non-queueable with fresh bleed telemetry, source-identity trust, chatter guard, hysteresis evidence, and rollback/readback requirements. | separate ADR after WING MVP |
| FX / Sends | proposal_only | Keep FX/send proposals non-queueable with fresh send/return context, mapped FX schema hints, return-level safety, monitor-safety guard, feedback/intelligibility limits, and rollback/readback requirements. | separate ADR after WING MVP |
| Routing / Groups | read_only_map | Keep routing/group/DCA/matrix topology maps read-only with fresh snapshot coverage, destructive-change markers, membership evidence, and separate non-queueable proposal plans. | separate ADR after WING MVP |
| Scenes / Presets | read_only_inventory | Keep scene/preset inventories read-only with fresh catalog snapshots, load/save policy markers, scoped diff/rollback evidence, and separate non-queueable proposal plans. | separate ADR after WING MVP |
| Master / Output | proposal_only | Keep master/output recommendations non-queueable with true-peak headroom, PA-feed safety, reference-feed exclusion, fresh main-bus/readback evidence, limiter/dynamics proof, and rollback requirements. | separate ADR after WING MVP |

## Selection Rule

Codex may work on this backlog only when one of these is true:

1. the work directly supports the first non-closed WING MVP blocker;
2. the WING hardware is unavailable and the work stays offline/proposal-only;
3. the user explicitly asks for a method-family design or implementation.

Even then: Do not widen live writes.
