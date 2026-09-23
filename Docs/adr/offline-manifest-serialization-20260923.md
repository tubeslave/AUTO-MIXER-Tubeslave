# Native JSON boolean in the offline pipeline manifest

## Context

The canonical REPA wrapper completed a real eight-track render but exited 1
while serializing `source_envelopes.*.crest_ok`. The value comes from a NumPy
comparison. A complete WAV did not mean a successful pipeline result.

## Options considered

Normalize at the manifest boundary, alter the underlying DSP value types, or
use a generic JSON encoder/default hook.

## Decision

Convert `crest_ok` to a native bool when building the existing manifest DTO.

## Why this won

It preserves the field type and meaning, avoids changing DSP, and directly fixes
the observed CLI failure. Pre-fix and post-fix drum WAV hashes are identical.

## Rejected alternatives

`default=str` would silently change the result type; a broad encoder or DSP
change is unnecessary for this concrete export boundary.

## Implementation plan

One conversion in `build_source_envelopes`; regression using actual WAV
extraction for silence, a tone and noise, covering false and true results.

## Test plan

Run the focused envelope/pipeline tests and canonical CLI on the two available
multitracks, inspect WAV integrity and true peak, then run the standard suite.
CP05 remains unbound following the previous source change; this is not new
hardware evidence. No independent review or listening acceptance is claimed.

## Risks and rollback

The render is unchanged. Revert the one-line conversion to restore the prior
CLI behavior, including its serialization failure on active material.
