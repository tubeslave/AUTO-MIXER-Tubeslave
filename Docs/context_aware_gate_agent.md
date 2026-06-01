# Context-Aware Gate / Bleed Agent

## Purpose

`ContextAwareGateAgent` is a dry-run recommendation layer for gate and expander
settings. It is built around the ACAD-style idea that noise depends on context:
cymbals in an overhead channel are useful signal, but cymbals in a vocal mic are
usually bleed.

The module does not control the console. It produces operator-readable
recommendations and decision-trace payloads only.

## Inputs

- `channel_id`
- `channel_label`
- `expected_instrument`
- channel audio or an existing `ChannelFeatureSnapshot`
- optional peer channel audio for bleed/correlation estimation

## BleedTelemetry

The agent emits these metrics:

- `noise_floor_db`
- `activity_ratio`
- `bleed_correlation_score`
- `spectral_outlier_score`
- `transient_density`
- `tail_risk`
- `chatter_risk`
- `false_open_risk`
- `false_close_risk`

## Output Contract

Every recommendation is dry-run:

- `agent_name = ContextAwareGateAgent`
- `action_type = gate | expander`
- `suggested_params`
- `confidence`
- `risk_score`
- `reason`
- `safety_status`
- `dry_run = true`
- `idempotency_key`
- `evidence.telemetry`

## Safety

- No WING/OSC transport imports.
- No live gate/expander writes.
- Gate/expander output remains proposal-only.
- High-risk recommendations stay review-required.
- Decision traces use `proof_status = dry_run_only` and
  `blocked_reason = gate_expander_live_write_quarantined`.

## Current Integration

- `backend/context_aware_gate.py` defines telemetry, agent logic, and trace conversion.
- `backend/agentic_method_foundation.py` now uses it for gate recommendations.
- `backend/processing_registry.py` exposes `context_aware_gate_agent` as
  `proposal_only`.
- `config/automixer.yaml` and `config/default_config.json` keep it disabled and
  dry-run by default.

## Next Steps

1. Feed real channel telemetry and offline bleed simulations into the agent.
2. Surface `BleedTelemetry` in the frontend Channel Intelligence / Safety panels.
3. Store accepted/rejected recommendations in the learning telemetry pipeline.
4. Keep live gate/expander writes blocked until a separate supervised hardware
   proof exists.
