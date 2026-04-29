# Cleanup Candidates

No code was deleted for this task. The following items are candidates for a
future cleanup pass only after a separate review and tests.

## Large overlapping offline mixers

- `tools/offline_agent_mix.py`
- `tools/chat_only_shared_mix.py`
- `src/experiments/muq_eval_director.py`

Reason: these overlap in offline rendering, channel plans, reports, and
candidate evaluation. They are still useful references and may contain behavior
not covered by the new chain, so they should not be removed now.

## External research directories

- `external/automix-toolkit`
- `external/FxNorm-automix`
- `external/Diff-MST`
- `external/MuQ-Eval`
- `external/dasp-pytorch`

Reason: they are large and partly duplicated in purpose, but they document the
research base requested by the project. Keep until the adapters have replaced
specific use cases and licensing/weight requirements are clear.

## Legacy direct hardware scripts

- `backend/reset_all_channels.py`
- `backend/reset_modules_trim_faders.py`
- `backend/disable_modules_set_faders.py`
- `backend/load_snap*.py`

Reason: these can perform destructive mixer operations and should remain out of
offline tests. They may still be operator tools, so cleanup requires a dedicated
hardware-safety review.

## Duplicate or broad audio I/O fallbacks

- Direct PyAudio fallbacks in older modules such as `backend/auto_eq.py`,
  `backend/lufs_gain_staging.py`, and voice-control modules.

Reason: `backend/audio_capture.py` is the canonical capture service, but these
fallbacks may still support legacy workflows. Do not delete without confirming
the current live startup paths.
