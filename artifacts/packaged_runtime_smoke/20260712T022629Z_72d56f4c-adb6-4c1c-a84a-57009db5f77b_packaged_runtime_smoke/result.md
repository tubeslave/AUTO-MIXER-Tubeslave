# Packaged runtime smoke

- Overall: FAIL
- Started packaged app: PASS
- Secure IPC bootstrap: FAIL
- Invalid or expired token rejected: FAIL
- Fresh token recovery authenticated: FAIL
- Immutable offline lock reported: FAIL
- Mixer connect/discovery operations blocked: FAIL
- Stale session token removed on reload: FAIL
- Hardware controls hidden: FAIL
- OS-enforced non-loopback network deny: PASS
- Launched copy matches signed app before/after: PASS
- External process-group connections: 0
- Continuous network samples: 105
- WING-port socket observations: 0
- Canonical WING log unchanged: PASS
- Console WAL unchanged: PASS
- Complete application tree unchanged: PASS
- Temporary write artifacts absent: PASS
- Source binding: dirty_tree_non_clean_checkout

This artifact proves an offline packaged runtime launch only. It grants no live authority and is not macOS notarization evidence.
