# Automixer Port Map From smartEQ4 Observations

This is a design map, not a clone of Sonible smartEQ4. It translates observable
architecture into safe Automixer-owned behavior.

## Current Automixer Fit

The current `offline_smarteq_eq` branch already matches several observed
signals:

- 24 EQ slots via `OfflineSmartEQConfig.band_slots = 24`
- FFT/LTAS analysis
- role-specific target curves
- smart-weight-style correction regions
- protected fundamental ranges
- conservative cut/boost caps
- offline-only rendering and JSON reports

The current gap is that the method is mostly static EQ. The observed smartEQ4
surface includes dynamic band controls, group impact, analyzer state, auto-gain
state, and profile-module families.

## Portable Concepts

| Observed smartEQ4 surface | Automixer-owned equivalent |
| --- | --- |
| `eq_band00` through `eq_band23` | 24-slot proposal schema for offline/shadow dynamic EQ |
| `frequency`, `gain`, `quality`, `type` | existing `EQBand` / `ParametricEQBand` fields |
| `dynamic`, `threshold`, `ratio`, `attack`, `release`, `dynamic_gain_range` | extend current dynamic EQ playbook schema from 6-band external predictions to 24-slot internal proposals |
| `eq_smartweight0/1` | keep current smart-weight regions, add explicit `mode`, `slope_hi`, `slope_lo`, and `adaptive` fields in reports |
| `learner_profile_id` and `.nn` profile families | replace proprietary profiles with Automixer role families: vocal high/low, speech high/low, bass, drums, hihat, snare, piano, acoustic guitar, bowed low strings, universal |
| `xcf_group_enabled`, `xcf_group_impact` | group-aware correction pressure derived from bus/stem context; cut masking regions before boosting |
| `general_auto_gain_enabled`, `general_auto_gain_mode` | keep auto gain separate from EQ and bounded by current input-trim/headroom policy |
| `eq_use_linear_phase` | offline-only optional linear-phase render flag; live remains minimum-phase/proposal-only |
| analyzer controls | report analyzer window/range/speed metadata; do not expose as live writes |

## Proposed Automixer Method Boundary

Method id:

```text
automixer_profile_dynamic_eq_shadow_v1
```

Safe execution modes:

- offline render: can apply to rendered stems with peak and loudness checks
- soundcheck: proposal-only; operator can approve bounded EQ changes later
- live: shadow/audit-only until supervised hardware proof exists

Live exclusions:

- no direct WING EQ writes
- no OSC apply/reset path
- no automatic fader or trim boost as part of EQ
- no use of proprietary `.nn` files or Sonible model output

## Implementation Shape

1. Add an internal 24-slot dynamic EQ proposal type.
2. Reuse `backend/dynamic_eq_playbook.py` validation bounds for threshold,
   ratio, attack, release, dynamic range, Q, and gain.
3. Add profile families using Automixer-owned target curves, not Sonible model
   files.
4. Add group-impact scoring from existing channel role and bus/stem context.
5. Keep `offline_smarteq_eq.method` as the offline renderer, but enrich reports
   with dynamic fields and smart-weight metadata.
6. Add focused synthetic tests:
   - 24-slot schema parses and validates
   - excessive dynamic gain range is rejected
   - group-impact prefers cuts over boosts in masking bands
   - live mode returns proposal-only records
   - offline mode still respects final peak ceiling

## Practical Next Step

Do not try to import smartEQ4 internals. The next safe code step is to extend
Automixer's current `offline_smarteq_eq` and `dynamic_eq_playbook` boundary so
the project can represent a 24-slot, profile-aware, group-aware dynamic EQ
proposal while preserving the existing live-safety gate.
