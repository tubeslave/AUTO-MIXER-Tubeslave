# ART DSP Canvas reference analysis

Date: 2026-08-04
Method: official public pages plus read-only package trust/manifest inspection
Installation/launch: not performed

## Product mechanics confirmed from public sources

- Canvas visualizes tracks as objects/blobs and allows spatial manipulation of
  mix attributes.
- ART DSP defines left/right as panning, up/down as frequency and back/front as
  volume.
- ART DSP explicitly says Canvas is not a 3D binaural panner.
- It provides per-instrument and full-mix presets, built-in EQ and spectral
  sidechaining.
- Same-format plug-in instances can communicate for the multi-track view.

Sources:

- <https://www.artdsp.net/canvas>
- <https://artdsp.net/support>

## Supplied package evidence

File: `/Users/dmitrijvolkov/Downloads/Canvas-v1.0.0.pkg`
SHA-256: `dcab188a2a55d8eca3aac263080b25fb47d12c695ba15a9fbdefd07fdf70f3d3`

- Version 1.0.0; universal Apple Silicon/Intel payloads.
- Contains AU and VST3 plug-ins for the standard system plug-in directories.
- Developer ID installer signature: `DRU LTD. (EWZB736XZ9)`.
- macOS verification reported an accepted notarized Developer ID package.
- No installer scripts were present in the package manifest.

This establishes package identity and installation surface only. It does not
establish audio quality, runtime safety or suitability as an Automixer
dependency. The package was not installed. No proprietary implementation or
asset is reused.

## Automixer interpretation

Adopt the public perceptual mapping as a visualization vocabulary, with stricter
evidence rules:

- X = console pan readback.
- Y = measured spectral centroid/brightness, never an instrument-name preset.
- Z = measured level/loudness.
- Blob identity = Automixer channel/source identity and provenance.
- Missing axes remain missing.
- The first product slice is inspect-only and cannot generate console actions.

This is a mix-space visualization, not a claim that Automixer knows an
instrument's physical location in the room.
