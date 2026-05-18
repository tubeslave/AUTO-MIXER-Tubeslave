# Backend Manual Probes And Legacy Checks

Date: 2026-05-18
Status: structured inventory, not runtime approval

These scripts are not part of the canonical backend test suite under `tests/`.
They are manual checks, routing probes, lab tools, or older one-off tests. They
remain in `backend/` for now because many assume imports and working directory
relative to that folder.

## Routing And WING Probes

- `backend/auto_test_routing.py`
- `backend/check_channel_10.py`
- `backend/check_output_routing.py`
- `backend/check_test_ready.py`
- `backend/find_routing_addresses.py`
- `backend/find_routing_outputs_addresses.py`
- `backend/get_channel_node.py`
- `backend/list_all_snapshots.py`
- `backend/list_snapshots.py`
- `backend/monitor_osc_routing.py`
- `backend/query_channel1_routing.py`
- `backend/query_snap_info.py`
- `backend/route_channel_inputs.py`
- `backend/route_channels.py`
- `backend/route_custom_outputs.py`
- `backend/route_dante_outputs.py`
- `backend/test_card_routing.py`
- `backend/test_channel2.py`
- `backend/test_routing_set.py`
- `backend/test_scan_mixer_websocket.py`
- `backend/test_snap_recall.py`
- `backend/test_wing_connection.py`

## Gain, Trim, And DSP Probes

- `backend/test_filter_eq.py`
- `backend/test_filter_eq_correct.py`
- `backend/test_filter_eq_final.py`
- `backend/test_filter_eq_simple.py`
- `backend/test_gain_staging_complete.py`
- `backend/test_gain_staging_dante.py`
- `backend/test_gain_staging_full.py`
- `backend/test_gain_staging_realtime.py`
- `backend/test_modules.py`
- `backend/test_trim.py`
- `backend/test_trim_multiple.py`
- `backend/test_trim_reset.py`

## Voice Probes

- `backend/test_voice_button.py`
- `backend/test_voice_control.py`
- `backend/test_voice_integration.py`
- `backend/test_voice_simple.py`

## Backup And Restore Tools

- `backend/backup_channels.py`
- `backend/restore_channels.py`

## Policy

- Do not import these scripts into live runtime.
- Do not use them as release proof. Release proof belongs in `tests/`.
- If a probe becomes important, convert it into a focused test under `tests/`
  or move it into `backend/lab_only/` with explicit setup notes.
- If a probe can write to WING, treat it as unsafe unless it is proven to use
  the supervised gate.
