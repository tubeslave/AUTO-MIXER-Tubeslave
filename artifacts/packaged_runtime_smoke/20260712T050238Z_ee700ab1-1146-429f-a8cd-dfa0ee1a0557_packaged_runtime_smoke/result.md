# Packaged runtime smoke

- Overall: FAIL
- Started packaged app: FAIL
- Backend ready within unknown ms: FAIL
- Secure IPC bootstrap: FAIL
- Invalid or expired token rejected: FAIL
- Fresh token recovery authenticated: FAIL
- Immutable offline lock reported: FAIL
- Mixer connect/discovery operations blocked: FAIL
- Stale session token removed on reload: FAIL
- Hardware controls hidden: FAIL
- OS-enforced backend non-loopback network deny: FAIL
- Launched copy matches signed app before/after: FAIL
- External process-group connections: unknown
- Continuous network samples: unknown
- WING-port socket observations: unknown
- Canonical WING log unchanged: PASS
- Canonical decision trace log unchanged: PASS
- Canonical decision memory log unchanged: PASS
- Console WAL unchanged: PASS
- Complete application tree unchanged: PASS
- Backend onedir bundle unchanged: PASS
- Temporary write artifacts absent: PASS
- Source binding: dirty_tree_non_clean_checkout
- Build profile: local
- Release eligible: false
- Build profile bound to app: PASS

This artifact proves an offline packaged runtime launch only. It grants no live authority and is not macOS notarization evidence.
