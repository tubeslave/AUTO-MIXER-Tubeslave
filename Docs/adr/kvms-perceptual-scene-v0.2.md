# ADR-KVMS-002 — отдельный offline/shadow слой оценки музыкальной сцены

Date: 2026-11-18  
Status: PROPOSED / research only / NOT MERGED

## Context

Existing `Docs/perceptual_evaluation.md` defines MERT/lightweight embedding-based pre/post shadow evaluator. No reliable calibrated relationship is established between a raw embedding delta and human recognition of important musical roles. BleedInfo in `backend/auto_fader_v2/core/bleed_detector.py` does not certify source attribution.

## Options considered

1. Directly optimize existing embedding score in live Decision Engine — **rejected**, lacks listener calibration and post-console certainty.
2. Replace live spectral masking system with musical role model — **rejected**, unverified and unsafe.
3. Add offline/shadow `KVMSProxy` with role and attribution gates — **chosen**.

## Decision and rationale

Build standalone deterministic baseline using aligned dry stems; require explicit `foreground` role and verified `TARGET` attribution; output immutable offline proposal records without applying gain. Reserve intentional blend/support for future controlled research rather than forcing separation.

## Implementation

`kvms_shadow.py` + `test_kvms_shadow.py`, isolated. No edits to `AutoSoundcheckEngine`, configs, mixer clients or `perceptual` evaluator. Future integration must explicitly negotiate typed proposal schema and evidence provenance.

## Validation

Run `python -m pytest -q test_kvms_shadow.py` (current run 15 passed). Need source-disjoint independent listening studies before product decisions. Run dry/mocked integration regression tests before any runtime modification.

## Risks

False source attribution, bleed, mismatch between STFT overlap and human hearing, amplitude bias, changing section role, missing reference, overreacting to synthetic probes, production latency/CPU, stereo/phase artifact invisibility.

## Rollback

Do not load standalone shadow component. No live configuration or console state modified; delete code package without migration. If ever integrated, disable KVMS flag and verify live behavior unchanged.