# M110 Council decision

The Council accepts a two-stage provenance boundary and four separate one-shot
authority kinds.

1. `capture_wing_master_route_proof` is authenticated, fixed-address,
   observation-only and process-private; its artifact is audit evidence only,
   and restart requires a new proof. Capture is bound to one unique physical
   BEHRINGER WING CoreAudio/USB VID/PID/location/serial identity and exact
   PortAudio stream, not merely a device name.
2. Exact capabilities are backend-generated and atomically consumed behind the
   canonical `apply_operator_proposal` boundary.
   The same route-proof owner arms supervised capability-only mode through an
   exact authenticated message; restart/reconnect revokes and requires reproof.
3. Restore-only proof ends as `proof_passed` / `proof_restored` and grants no
   normal authority.
4. Capabilities are issued only to the authenticated
   `m110_attended_runner` launch session by a strict v1 request, with one token
   per owner, eight globally, 30-second latest-start TTL and disconnect revoke.
   The session is established by an exact prestart source manifest plus random
   secret over a one-shot inherited FD; route capture, arm and issuance require
   fresh-nonce HMACs, never self-declared environment provenance.
5. DSP reset, atomic processor-parameters-plus-enable and staged child commits
   are distinct commit-once lanes with separate approvals. Each permanent
   operation requires a recent proof with the identical canonical operation
   bundle digest. Reset and processor enable have fixed composite bundles;
   staged children are decomposed into single-operation bundles. No family
   sibling inference is allowed.
6. Reset preservation means finite-manifest physical equality plus
   no-outside-send evidence and reporting only transition events actually
   delivered by WING for unmeasured categories, not an unsupported claim that
   event silence or an incomplete stream proves every console surface equal.
7. The normal allowlist and registry are not widened by M110.
8. No real WING contact occurs until Council artifacts, implementation,
   adversarial tests, full-suite/no-WING validation, clean source commit and
   independent review are complete. A backend-owned CP05 gate binds those
   results to every entry in the committed Git tree; a later commit closes the
   gate. A centralized attended-child ingress allowlist blocks raw capture,
   discovery and alternate-mixer handlers before resource construction while
   preserving stop/disconnect/emergency cleanup.

Rejected: direct runner OSC, generic query/write endpoints, self-authored route
JSON, public proof flags, broad legacy reset and automatic family promotion.
