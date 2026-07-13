import assert from 'node:assert/strict';
import test from 'node:test';

import {
  classifyApplyToneValue,
  classifyReadbackToneValue,
  classifyRollbackToneValue
} from '../operatorStatusTone.mjs';

test('readback tones fail closed for every non-success terminal state', () => {
  const cases = [
    ['verified_from_physical_readback', 'green'],
    ['verified_from_cache', 'yellow'],
    ['pending', 'yellow'],
    ['not_verified', 'red'],
    ['readback_cache_not_fresh', 'red'],
    ['physical_readback_mismatch', 'red'],
    ['', 'red'],
    ['unknown_backend_status', 'red']
  ];
  cases.forEach(([status, expected]) => {
    assert.equal(classifyReadbackToneValue(status), expected, status);
  });
});
test('rollback tones fail closed for CAS, cancellation, and unknown states', () => {
  const cases = [
    ['rollback_physical_verified', 'green'],
    ['rollback available', 'yellow'],
    ['not_attempted', 'yellow'],
    ['rollback_cas_target_changed', 'red'],
    ['rollback_unverified', 'red'],
    ['rollback_cancelled', 'red'],
    ['manual_review_required', 'red'],
    ['unknown_backend_status', 'red']
  ];
  cases.forEach(([status, expected]) => {
    assert.equal(classifyRollbackToneValue(status), expected, status);
  });
});

test('apply tones only allow explicit applied or in-progress states', () => {
  const cases = [
    ['applied', '', false, 'green'],
    ['pending', '', false, 'yellow'],
    ['apply_error', '', false, 'red'],
    ['manual_review_required', '', false, 'red'],
    ['applied', 'manual_review_required', false, 'red'],
    ['applied', '', true, 'red'],
    ['unknown_backend_status', '', false, 'red']
  ];
  cases.forEach(([status, reason, rollbackFailed, expected]) => {
    assert.equal(
      classifyApplyToneValue(status, reason, rollbackFailed),
      expected,
      `${status}:${reason}`
    );
  });
});
