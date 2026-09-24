# WING lab transport repair

Operator request: repair batch application, then conduct an attended real-console
soundcheck. This patch is the software prerequisite, not hardware authorization.

Observed defects: the lab executor never installs the mandatory WingClient
mutation boundary; its fake transport tests miss that integration. A send or WAL
failure after packet emission can also omit the operation from the rollback stack.

Scope: backend/wing_lab_autowrite.py and focused transport-integration tests.
No DSP changes, no CP05 evidence rewrites, no console network access during tests.
Keep production allowlists and lab allowed families unchanged. A successful unit
test is not proof of firmware mapping, physical rollback, or acoustic improvement.

Proposal: an exclusive deny-by-default transport boundary allowing only the exact
journalled active operation, held through queued-send draining; serialized session
lifecycle; rollback preimage registered before the first possible packet; failed
recovery remains explicit and retryable. Deny sends after expiry except restoration
of this session's recorded preimages. Verify cleanup and re-arm failure paths.

Tests: real WingClient.send with in-memory datagram sink, explicit deny tests,
throttled flush, failed send/WAL, rollback retry, expiry and session teardown.
Independent review is required before any hardware stage. Existing untracked
offline diagnostic artifacts are unrelated and must not be staged or overwritten.
