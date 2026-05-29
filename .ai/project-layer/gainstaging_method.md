# Gain-Staging Method

## Purpose

This file records the current Automixer input-trim and balance method so agents
do not confuse method work with infrastructure work.

## Canonical Path

The canonical automatic input-trim path is `input_trim_preamp_character` through
`SignalStats.calculate_input_trim_recommendation(...)`. Treat older LUFS-only
helpers as legacy context unless current code proves they are active.

## Method Boundary

This method is about input trim and preamp gain structure. It must not drift into
unapproved fader automation, bus-balance execution, FX choices, routing, or
autonomous live control.

## Soundcheck Behavior

During soundcheck, Automixer may analyze longer windows, source character,
headroom, transient shape, bleed risk, and Ayaic level planes. It should produce
clear recommendations and dry-run evidence before any supervised change.

## Live Behavior

During live operation, the method is conservative. It should reduce overload
risk, avoid boosting from short or bleed-heavy windows, respect true-peak and
feedback safety, and require explicit opt-in before any gain increase.

## Ayaic Level Planes

Ayaic Mix Monolith level planes are the default balance reference for offline
mixes, soundchecks, and live mode. Measure individual sources and the summed
stems or groups they form. Safety limits override target loudness.

## Required Evidence

Method changes should include focused tests for input trim behavior and broader
smoke coverage when the change affects operator recommendations, server DTOs, or
proof/dry-run runners.
