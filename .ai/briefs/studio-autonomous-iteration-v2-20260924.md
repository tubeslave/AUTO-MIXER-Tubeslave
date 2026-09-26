# STUDIO Autonomous Iteration v2 — 2026-09-24

## Goal

Connect one bounded Perceptual Critic hypothesis to the real offline mastering delivery path while preserving immutable baseline identity and keeping subjective acceptance human-only.

## Repository evidence

The existing autonomous loop already distinguishes `machine_safe`, `pending_human_review` and `rejected`, and the offline mastering facade already validates WAV/MP3 delivery. The missing boundary is persistent orchestration between those two systems: candidate/baseline hashes, explicit rollback source selection, immutable evidence, and a guarantee that an autonomous run cannot promote a subjective audio baseline.

## Bounded task

- add a STUDIO-only file-oriented iteration runner;
- require a causal plan with a no-change/bypass counterfactual;
- require baseline/candidate sample-rate and shape identity;
- persist file and decoded-audio hashes for both baseline and candidate;
- route Perceptual Critic rejection to the baseline before mastering;
- route machine-safe/uncertain audible candidates to mastering only as listening candidates;
- propagate mastering rejection without promoting the candidate;
- refuse existing output directories and verify both inputs remain byte-identical;
- never accept a human-review decision as an autonomous input;
- never touch live console, OSC or realtime code.

## Acceptance

Focused CI must prove: pending candidate -> audition delivery with unchanged baseline; critic rejection -> baseline rollback before mastering; mastering rejection -> no promotion; misaligned files fail before delivery; no-change counterfactual is mandatory; one synthetic candidate crosses the real `deliver_master()` WAV/MP3 boundary. Full repository CI must remain green on Python 3.10/3.11/3.12.

## Human listening

This boundary may produce an audition master. It cannot make that audition the new artistic baseline. Subjective acceptance remains an explicit human listening action outside this autonomous runner.
