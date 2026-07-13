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
- External process-group connections: unknown
- Canonical WING log unchanged: PASS
- Console WAL unchanged: PASS
- Source binding: dirty_tree_non_clean_checkout

This artifact proves an offline packaged runtime launch only. It grants no live authority and is not macOS notarization evidence.
