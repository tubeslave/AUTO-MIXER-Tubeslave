# dLive read-only connection runbook

## Current boundary

This runbook creates no live-write authority. The normal backend and factory
instantiate `DLiveClient` read-only. A TCP-open result is a candidate endpoint,
not proof of model, firmware, show, patch, or ownership.

## 1. Physical and network checks

1. Confirm whether the cable terminates at the **Surface** or **MixRack**
   Network port. Do not guess from the port number.
2. On dLive, record the exact IPv4 address, firmware (`V2.0+` required), MIDI
   base channel from `Utility / Control / MIDI`, and whether TLS is enabled.
3. Put only the dedicated Mac Ethernet interface in the same unique `/24`.
   Leave its router blank. Do not reuse Wi-Fi/VPN as the console route.
4. Before choosing a Mac address, verify that it is unused. Never duplicate the
   Surface or MixRack address.
5. Keep PA outputs contained and an operator at the console even though this
   checkpoint is read-only.

Observed on 2026-08-03 after network correction: `en7` is active at
`192.168.3.59/24`, and the route to `192.168.3.70` resolves directly through
`en7`. TCP `51325` is open while `51327/51328/51329` are closed, identifying a
plain-TCP MixRack endpoint candidate. A bounded read-only base-channel search
found MIDI base `11` (screen channel 12); input 1 returned two fresh stable
name replies `KICK`. A subsequent double read returned `KICK`, approximately
`-0.58 dB`, mute off, with clean disconnect and zero mutation commands. These
facts do not replace operator attestation of model, firmware, loaded show or a
physical Surface comparison.

Canonical observed artifact on 2026-08-04: the operator attested MixRack
`CDM32`, Surface `C3500`, firmware `2.0` and Show `lv 1 master`. The runner
again observed input 1 as `KICK`, approximately `-0.58 dB`, mute off, with two
fresh replies per field, six GETs, zero mutation writes and clean disconnect.
Artifact:
`/Users/dmitrijvolkov/.automixer/runtime/dlive_readonly_probes/20260804T085345Z_04849741-00f0-4d09-8d86-534df1415268.json`;
SHA-256 `f38220bdf85aa606c491ba8ae9323e3db8f2da34ca23bbc356b2161288c93be8`.
The operator subsequently confirmed all three values on the physical C3500
display: `KICK`, `-0.5827 dB`, mute off. The SHA-bound read-only artifact is
accepted; this grants no mutation or standing live authority.

## 2. Validate config without transport

Surface, plain TCP:

```bash
python tools/dlive_no_transport_preflight.py \
  --host 192.168.3.71 --endpoint surface --port 51328
```

MixRack, plain TCP:

```bash
python tools/dlive_no_transport_preflight.py \
  --host 192.168.3.70 --endpoint mixrack --port 51325
```

The receipt must say `console_contacted=false`, `midi_sent=false`,
`live_write=false`, and `live_authority_granted=false`.

## 3. Read-only attended proof

After the route points through the dedicated Ethernet interface:

1. Check only the endpoint-specific TCP port: MixRack `51325/51327`, Surface
   `51328/51329`.
2. Have the operator attest endpoint, IP, model, firmware and loaded show.
3. Connect with normal backend authority (read-only).
4. Query one known input's name, mute and fader twice. Each response must be
   fresh, stable and match the physical Surface display.
5. Stop on timeout, disconnect, unstable value, selector mismatch, unexpected
   traffic, or any identity discrepancy.

Canonical runner (replace every attested value with the Surface display):

```bash
python tools/dlive_attended_readonly_probe.py \
  --operator-attended --ack-read-only \
  --host 192.168.3.71 --endpoint surface --port 51328 \
  --expected-interface en7 \
  --midi-base-channel 11 --user-profile 0 --input-channel 1 \
  --model CDM32+C3500 --firmware 2.0 --show 'lv 1 master' \
  --expected-name KICK
```

The artifact is written outside the repository under
`~/.automixer/runtime/dlive_readonly_probes/`. Operator identity fields remain
attestations (`machine_verified=false`); the artifact proves only fresh stable
protocol readback and clean disconnect. The artifact also records the exact
`user_profile`; artifacts created before that field was added cannot authorize
the restore-only proof and must be refreshed.

## 4. One-parameter restore-only mutation proof

Do not run this from the normal connection path. First accept the read-only
artifact and keep the same exact endpoint, interface, show, MIDI base channel,
input and name. Physically contain all PA/monitor outputs and mute the selected
input on the dLive Surface. Choose a target below the fresh preimage, with a
maximum cut no greater than 6 dB.

The selected target must be finite and strictly below the fresh preimage after
7-bit MIDI quantization. Do not use `-54 dB` because it maps to `-inf`. If the
fresh preimage is above `0 dB`, this proof is not permitted: move it manually to
`0 dB` or below, repeat the read-only proof, and create a new approval.

Before requesting operator approval, validate the exact immutable plan without
route lookup or TCP:

```bash
python tools/dlive_validate_fader_proof_plan.py \
  --readonly-artifact /Users/OPERATOR/.automixer/runtime/dlive_readonly_probes/REPLACE.json \
  --readonly-artifact-sha256 REPLACE_64_HEX_SHA256 \
  --approval-id REPLACE_UUID_OR_OMIT_TO_GENERATE \
  --host 192.168.3.71 --endpoint surface --port 51328 \
  --midi-base-channel 11 --user-profile 0 --expected-interface en7 \
  --input-channel 1 --expected-name KICK \
  --model CDM32+C3500 --firmware 2.0 --show 'lv 1 master' \
  --target-db -3.0 --max-cut-db 3.0
```

Only `status=ready_for_operator_confirmation` is acceptable. The receipt must
also say `route_checked=false`, `console_contacted=false`, `midi_sent=false`,
`execution_allowed=false` and `live_authority_granted=false`. It prints the
exact confirmation token and both apply/rollback packet hashes; it never
records physical containment, Receive MIDI or operator approval.

The runner performs only:

`fresh preimage -> WAL boundary -> downward apply -> two fresh readbacks -> CAS -> WAL boundary -> exact rollback -> two fresh readbacks -> revoke/disconnect`

Hardware note from 2026-08-04: one pre-pairing attempt using a MIDI-valid
full-status NRPN form returned the unchanged preimage. AUTO-MIXER subsequently
aligned its encoder with the compact documented running-status shape
`BN 63 CH 62 17 06 LV`. Do not interpret the first no-op as proof that
full-status MIDI is intrinsically rejected: effective Receive MIDI/security,
endpoint state and wire delivery were not independently established then.
The attempt is terminal and its approval is consumed.

The compact packet was also ignored by the pre-pairing MixRack endpoint. Treat
`Receive MIDI` as an explicit physical gate: verify it is enabled in
`Utility / Control / MIDI` and pass `--receive-midi-confirmed-enabled`; the
runner blocks before hardware contact when that attestation is absent. For the
next attended physical-fader proof use the independently GET-verified C3500
Surface endpoint `192.168.3.71:51328`, not the prior MixRack endpoint.

After Surface/MixRack pairing was machine-observed in Director, fresh GET-only
probes against both endpoints agreed on `KICK`, raw fader `106` and Mute ON.
One newly approved Surface attempt sent `BB 63 00 62 17 06 65`, but four fresh
readbacks and the operator's physical observation stayed at raw `106`. Receipt:
`20260804T141746Z_def89219-2b15-4f4d-a1ef-6309e6bdec8e.json`, SHA-256
`f84d8b0394cfb428a5b4361e7ccca5dca78c3a34083e7cecd9bf8f2b523394f6`.
It is terminal with `write_state_unknown=false`, clean teardown and no standing
authority. Do not retry this same experiment. Before another mutation, verify
the endpoint security level (`No Security` for plain TCP, or use the TLS port
and credentials), reboot the affected Surface/MixRack if security was changed,
and obtain known-good-client or packet-capture evidence that distinguishes
console configuration from AUTO-MIXER transport.

### Offline wire-evidence intake

AUTO-MIXER does not start packet capture. An operator must supply a JSON capture
with schema `automixer.dlive_midi_capture.v1`, a known-good Allen & Heath client,
direction `client_to_selected_endpoint`, observation point
`selected_endpoint_ingress`, exact endpoint identity and the observed bytes.
Review it offline and save the accepted report inside the private runtime root:

```bash
python tools/dlive_review_midi_capture.py \
  --capture /ABSOLUTE/OPERATOR/CAPTURE.json \
  --host 192.168.3.71 --endpoint surface --port 51328 \
  --midi-base-channel 11 --user-profile 0 \
  --input-channel 1 --target-db -3.0 \
  --output /Users/OPERATOR/.automixer/runtime/dlive_midi_capture_reviews/REPLACE.json
```

Only `status=accepted` and `exact_wire_match=true` may feed the next plan. This
proves exact ingress bytes only; the report deliberately keeps console
acceptance, physical movement, execution and live authority false.

The restore-only approval is cryptographically bound to one exact read-only
artifact. The file must be the owner-controlled regular file under
`~/.automixer/runtime/dlive_readonly_probes/`; its SHA-256, endpoint, port,
MIDI base, user profile, input identity, muted state, fader preimage, six GETs,
zero-write authority and clean teardown are checked before route lookup or TCP
connection. A fresh hardware readback must still match the artifact's exact
7-bit fader value before the WAL/apply boundary.

Example shape; replace every value from the accepted read-only artifact and
choose the bounded target only after seeing the current fader:

```bash
python tools/dlive_attended_fader_restore_proof.py \
  --operator-attended --execute-restore-only-proof \
  --pa-outputs-contained --input-confirmed-muted \
  --receive-midi-confirmed-enabled \
  --security-state-attested --security-mode no_security \
  --readonly-artifact /Users/OPERATOR/.automixer/runtime/dlive_readonly_probes/REPLACE.json \
  --readonly-artifact-sha256 REPLACE_64_HEX_SHA256 \
  --wire-evidence /Users/OPERATOR/.automixer/runtime/dlive_midi_capture_reviews/REPLACE.json \
  --wire-evidence-sha256 REPLACE_64_HEX_SHA256 \
  --confirmation 'APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF:sha256:REPLACE_BINDING' \
  --approval-id REPLACE_FROM_PLAN_VALIDATOR \
  --host 192.168.3.71 --endpoint surface --port 51328 \
  --expected-interface en7 --midi-base-channel 11 --user-profile 0 \
  --input-channel 1 --expected-name KICK \
  --model CDM32+C3500 --firmware 2.0 --show 'lv 1 master' \
  --target-db -3.0 --max-cut-db 3.0
```

If security was changed for this setup, also pass
`--security-changed-for-setup --reboot-after-security-change-confirmed` only
after the affected Surface/MixRack has actually rebooted and the effective
setting has been rechecked. Omitting the reboot confirmation blocks before
route lookup or TCP.

Normal `DLiveClient.set_fader()` and `set_mute()` remain zero-send. The runner
holds the same per-user writer lock as WING. Approval lives for at most five
minutes, one authority can issue only one apply and one exact rollback, and
the capability is revoked on every exit. A successful software test is not a
physical proof; preserve the artifact, WAL path and SHA for independent review.

The confirmation is not a reusable generic phrase. It is deterministically
bound to the exact endpoint, interface, MIDI base, user profile, channel/name,
model/firmware/show attestation, target/raw fader value, maximum cut and
read-only artifact path/SHA. After all physical flags are true, invoking the
runner with the old generic phrase exits before route lookup/TCP and prints the
required `...:sha256:<binding>` token. The operator must approve that exact
token and matching `approval_id` emitted by the zero-contact plan validator.
Changing any bound field or approval ID invalidates the token. Once a proof
reaches durable `prepared`, that approval ID is consumed in the WAL and cannot
start another proof; a retry needs a fresh plan, ID, token and operator
approval. The runner also enforces its separate five-minute runtime window.

## 5. Interrupted proof recovery

An incomplete dLive WAL transaction blocks every new proof. Do not delete or
edit the WAL. With the same endpoint still connected, outputs physically
contained and input confirmed muted, recover only the exact transaction:

```bash
python tools/dlive_recover_fader_restore.py \
  --operator-attended --recover-exact-preimage-only \
  --pa-outputs-contained --input-confirmed-muted \
  --transaction-id REPLACE_FROM_WAL_RECEIPT
```

Recovery never reapplies the target. If fresh readback already equals the WAL
preimage it records a zero-send terminal row. If it equals the WAL target it
may send the exact preimage once and verify twice. Any other value, name
mismatch, unmuted channel, route mismatch, timeout or disconnect is
`manual_review_required` and must not be overwritten.

Mute, preamp, 48 V, scene, routing and DSP mutations are not part of this proof
and remain blocked.
