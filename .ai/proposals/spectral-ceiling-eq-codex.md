1. Short thesis

Add spectral ceiling EQ as a proposal/merge layer. It should analyze smoothed
spectra, emit broad safe moves, and let the existing AutoFOH safety and offline
render paths decide whether those moves become audio changes.

2. What I understood

The goal is not a COS Pro clone. The useful public idea is a musical guide:
role-aware spectral slope, roll-off boundaries, broad zones, and foreground or
background intent. Existing rules remain authoritative, and the new layer must
explain why it acted or why it stood down.

3. Proposed solution

- Create `backend/heuristics/spectral_ceiling_eq.py` with dataclasses, profile
  loading, log-frequency smoothing, target curve generation, tilt estimation,
  zone analysis, vocal priority demasking, merge helpers, and human logs.
- Add `configs/spectral_ceiling_profiles.yaml` for role profiles and position
  offsets.
- Add `spectral_ceiling_eq` to `config/automixer.yaml`.
- In `AutoSoundcheckEngine._apply_eq`, compute existing adapted EQ first, then
  request a spectral proposal and merge compatible broad moves into the four
  console EQ bands before existing safety execution.
- In `mix_agent`, append conservative `MixAction` proposals from the same module
  so offline apply/render can use them.
- Add `automixer.tools.inspect_spectral_ceiling` as a debug entry point.

4. Likely files to touch

- `backend/heuristics/spectral_ceiling_eq.py`
- `backend/auto_soundcheck_engine.py`
- `mix_agent/agent/decision_loop.py`
- `config/automixer.yaml`
- `configs/spectral_ceiling_profiles.yaml`
- `automixer/tools/inspect_spectral_ceiling.py`
- `tests/test_spectral_ceiling_eq.py`
- `tests/test_mix_agent/test_offline_pipeline.py`
- `docs/spectral_ceiling_eq.md`
- `Docs/adr/spectral-ceiling-eq.md`

5. Alternatives considered

- Replace the existing EQ preset adaptation: rejected because the request and
  project safety rules require preserving known behavior.
- Add a full match EQ renderer: rejected as too aggressive and fragile.
- Only log recommendations: useful for dry-run, but insufficient for the
  approved offline and live pipelines when config enables apply.

6. Risks

- Four-band live consoles have limited room for additional EQ moves, so merge
  must collapse broad suggestions conservatively and skip overlaps.
- Role detection can be wrong; the module must require confidence before apply.
- Background instruments could become dull if tilt logic is too strong; boost
  and cut limits must stay small.

7. Test plan

- Unit tests for guides, smoothing/clamp/profile selection and role position.
- Unit tests for dry-run, vocal demasking and master ±1 dB limit.
- Regression tests for existing offline apply.
- Standard full test command before final report.

8. Where the other agent may disagree

Kimi may argue for a separate queued action stage after `_apply_eq`. I prefer
merging before the existing safety apply because it avoids rate-limit collisions
with the four-band EQ and keeps legacy preset adaptation intact.
