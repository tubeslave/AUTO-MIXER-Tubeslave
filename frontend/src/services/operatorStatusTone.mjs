function normalizeStatus(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ');
}
export function classifyReadbackToneValue(status) {
  const value = normalizeStatus(status);
  if (value === 'verified from physical readback' || value === 'physical readback verified') {
    return 'green';
  }
  if (value === 'verified from cache' || value.includes('waiting') || value.includes('pending')) {
    return 'yellow';
  }
  return 'red';
}

export function classifyRollbackToneValue(status) {
  const value = normalizeStatus(status);
  if (
    value === 'rollback executed'
    || value === 'rollback physical verified'
    || value === 'rolled back'
  ) {
    return 'green';
  }
  if (
    value === 'rollback available'
    || value === 'not attempted'
    || value.includes('waiting')
    || value.includes('pending')
  ) {
    return 'yellow';
  }
  return 'red';
}

export function classifyApplyToneValue(status, reason = '', rollbackFailed = false) {
  if (rollbackFailed) return 'red';
  const value = normalizeStatus(status);
  const normalizedReason = normalizeStatus(reason);
  if (value === 'applied' && !normalizedReason.includes('manual review')) return 'green';
  if (['pending', 'applying', 'queued', 'updated'].includes(value)) return 'yellow';
  return 'red';
}
