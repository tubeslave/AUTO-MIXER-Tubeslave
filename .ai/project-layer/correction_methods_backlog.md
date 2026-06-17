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
| Pan / Stereo Placement | proposal_only_needed | Build offline/proposal pan suggestions with mono/phase safety and no live apply. | separate ADR after WING MVP |
| EQ | proposal_only | Keep corrective EQ proposals non-queueable with fresh EQ-band identity, feedback-notch/manual-band protection, bounded gain/headroom, fresh phase context, source-identity trust, and rollback/readback requirements. | separate ADR after WING MVP |
| Compressor | proposal_only | Keep dynamics proposals non-queueable with GR estimate, true-peak bounded makeup gain, feedback safety, fresh dynamics context, source-identity trust, and rollback/readback requirements. | separate ADR after WING MVP |
| Gate / Expander | proposal_only_needed | Build gate proposals with source-identity freshness and chatter/hysteresis protection. | separate ADR after WING MVP |
| FX / Sends | proposal_only_needed | Build FX/send proposal plans with return-level and monitor-safety boundaries. | separate ADR after WING MVP |
| Routing / Groups | read_only_map_needed | Build read-only routing/group maps and proposal-only plans. | separate ADR after WING MVP |
| Scenes / Presets | read_only_inventory_needed | Keep scene/preset work read-only unless operator explicitly imports/loads manually. | separate ADR after WING MVP |
| Master / Output | offline_method_needed | Build offline/proposal output recommendations with true-peak and PA-feed safety. | separate ADR after WING MVP |

## Selection Rule

Codex may work on this backlog only when one of these is true:

1. the work directly supports the first non-closed WING MVP blocker;
2. the WING hardware is unavailable and the work stays offline/proposal-only;
3. the user explicitly asks for a method-family design or implementation.

Even then: Do not widen live writes.
