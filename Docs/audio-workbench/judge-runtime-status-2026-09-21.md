# Learned Judge Runtime Status — 2026-09-21

Calibration pack was executed in the current mixing runtime.

## Available
- PyTorch: available.
- librosa: available.
- FFmpeg: available.
- deterministic calibration sensors: level, tonal balance, upper-mid harshness proxy, stereo width and crest/dynamics all detected the controlled perturbations; identical-file catch matched exactly.

## Not available in this runtime
- Qwen2-Audio: transformers/model runtime not installed.
- Audiobox Aesthetics: package/weights not installed.
- MuQ-MuLan: package not installed; checkpoint remains research/non-commercial guarded.

## Authority decision
No learned observer receives perceptual decision authority in this runtime.
Deterministic sensors remain measurement evidence only.
Overall musical preference routes to blind human A/B until a learned observer actually runs and passes capability calibration.

This is intentionally fail-closed. Missing models are not simulated by an LLM verdict.
