# STUDIO compression v2: bounded DSP correction

User asks to finish the compression module, after Belye Stai without neural audio.
Base: audio-workbench-mcp-v0.1 @ 381f8fad1cf30e046f71ca38adea792fd359b46e.

## Reproduced defects
mixing/dynamics.py (blob f230fa4b72952cb96c9fb6f059c1014990af41f0): silence,
10 ms and anti-phase stereo all raise IndexError from empty active selections.
Compression uses centered Gaussian smoothing on 20 ms frames, not independently
controllable attack/release. Detector sums L/R before squaring. Reported spread
is inferred from frame gain rather than remeasured rendered PCM. The separate
Belye Stai recipe has a causal compressor, but is not a repository module.

## One bounded task
Provide shared stateful causal linked mono/stereo feed-forward compression;
wire it into dynamics.apply and separate rider/makeup from compressor GR.
Conventional DSP, no mandatory new dependency, no learned models or live imports.
Validate static curve, independent time constants, chunk equivalence, stereo
polarity, sidechain filtering, silence/short input, finite checks, GR limits,
and actual output PCM evidence. Real raw Belye Stai excerpts are audition tests,
not a replacement master. Full exact-branch CI required before code acceptance.

## Evidence and limits
42 new focused tests pass locally. Local full-suite collection is blocked by
missing pythonosc; that is not called a passing suite. Full CI pending.
No independent Kimi reviewer available: self-review must not be presented as one.
Do not change live/backend compression, mastering or existing accepted audio.
Human listening remains required, no baseline promotion or paid services.
