export const DECISION_QUEUE_CONTRACT_VERSION = 2;
export const DECISION_QUEUE_WIRE_SCHEMA = 'automixer.decision_queue.v2';
export const PROPOSAL_REVIEW_SCHEMA = 'automixer.operator_proposal.review_content.v2';
export const PROPOSAL_DETAIL_MEDIA_TYPE = 'application/vnd.automixer.operator-proposal-review+json;version=2';

const FIRST_PAGE_SIZE = 50;
const MAX_DETAIL_CHUNK_BYTES = 65_536;
const MAX_DETAIL_TOTAL_BYTES = 2_097_152;
const MAX_DETAIL_CHUNKS = MAX_DETAIL_TOTAL_BYTES / MAX_DETAIL_CHUNK_BYTES;
const WIRE_MAX_BYTES = 131_072;
const SHA256_HEX_PATTERN = /^[0-9a-f]{64}$/;
const SHA256_DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

function record(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

function requiredString(value, field) {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`decision_queue_invalid_${field}`);
  }
  return value;
}

function strictString(value, field) {
  if (typeof value !== 'string') {
    throw new Error(`decision_queue_invalid_${field}`);
  }
  return value;
}

function nullableString(value, field) {
  if (value === null) return null;
  return requiredString(value, field);
}

function finiteInteger(value, field, minimum = 0) {
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`decision_queue_invalid_${field}`);
  }
  return value;
}

function strictBoolean(value, field) {
  if (typeof value !== 'boolean') {
    throw new Error(`decision_queue_invalid_${field}`);
  }
  return value;
}

function fixedCountMap(value, keys, field) {
  const source = record(value);
  const result = {};
  for (const key of keys) {
    result[key] = finiteInteger(source[key], `${field}_${key}`);
  }
  return result;
}

function normalizeProposalCurrentState(value) {
  const source = record(value);
  const blockerCodes = Array.isArray(source.apply_blocker_codes)
    ? source.apply_blocker_codes.map((code) => requiredString(
        code,
        'detail_current_state_apply_blocker_code'
      ))
    : (() => { throw new Error('decision_queue_invalid_detail_current_state_apply_blocker_codes'); })();
  const blockerCount = finiteInteger(
    source.apply_blocker_count,
    'detail_current_state_apply_blocker_count'
  );
  const expired = strictBoolean(source.expired, 'detail_current_state_expired');
  const canApply = strictBoolean(source.can_apply, 'detail_current_state_can_apply');
  if (blockerCount < blockerCodes.length || (canApply && (expired || blockerCount > 0))) {
    throw new Error('operator_proposal_detail_current_state_inconsistent');
  }
  return {
    status: requiredString(source.status, 'detail_current_state_status'),
    updated_at: nullableString(source.updated_at, 'detail_current_state_updated_at'),
    expired,
    can_apply: canApply,
    apply_blocker_codes: blockerCodes,
    apply_blocker_count: blockerCount
  };
}

function verifyWireEnvelope(value, field) {
  const wire = record(value);
  const maxBytes = finiteInteger(wire.max_bytes, `${field}_max_bytes`, 1);
  const serializedBytes = finiteInteger(
    wire.serialized_bytes,
    `${field}_serialized_bytes`,
    1
  );
  if (maxBytes !== WIRE_MAX_BYTES || serializedBytes > maxBytes) {
    throw new Error(`${field}_budget_invariant_breached`);
  }
}

export function isSha256Digest(value) {
  return typeof value === 'string' && SHA256_DIGEST_PATTERN.test(value);
}

export function buildDecisionQueueFirstPageRequest() {
  return {
    type: 'get_decision_queue',
    contract_version: DECISION_QUEUE_CONTRACT_VERSION,
    page_size: FIRST_PAGE_SIZE
  };
}

export function buildDecisionQueueNextPageRequest(cursor, expectedSnapshotRevision) {
  return {
    type: 'get_decision_queue',
    contract_version: DECISION_QUEUE_CONTRACT_VERSION,
    cursor: requiredString(cursor, 'cursor'),
    expected_snapshot_revision: requiredString(
      expectedSnapshotRevision,
      'expected_snapshot_revision'
    )
  };
}

function normalizeReviewIdentity(identity) {
  const source = record(identity);
  const proposalId = requiredString(source.proposalId, 'proposal_id');
  if (source.reviewContractVersion !== DECISION_QUEUE_CONTRACT_VERSION) {
    throw new Error('proposal_review_contract_version_unverified');
  }
  if (!isSha256Digest(source.reviewContentDigest)) {
    throw new Error('proposal_review_content_digest_unverified');
  }
  return {
    proposalId,
    reviewContractVersion: DECISION_QUEUE_CONTRACT_VERSION,
    reviewContentDigest: source.reviewContentDigest
  };
}

export function buildProposalDetailRequest(identity, chunkIndex = 0) {
  const verifiedIdentity = normalizeReviewIdentity(identity);
  return {
    type: 'get_operator_proposal_detail',
    contract_version: DECISION_QUEUE_CONTRACT_VERSION,
    proposal_id: verifiedIdentity.proposalId,
    expected_review_contract_version: verifiedIdentity.reviewContractVersion,
    expected_review_content_digest: verifiedIdentity.reviewContentDigest,
    chunk_index: finiteInteger(chunkIndex, 'detail_chunk_index')
  };
}

export function buildProposalApplyRequest({
  identity,
  approved,
  reason,
  approvalId,
  idempotencyKey
}) {
  const verifiedIdentity = normalizeReviewIdentity(identity);
  if (approved !== true) throw new Error('proposal_operator_approval_required');
  return {
    type: 'apply_operator_proposal',
    proposal_id: verifiedIdentity.proposalId,
    approved: true,
    reason: requiredString(reason, 'apply_reason'),
    approval_id: requiredString(approvalId, 'approval_id'),
    idempotency_key: requiredString(idempotencyKey, 'idempotency_key'),
    expected_review_contract_version: verifiedIdentity.reviewContractVersion,
    expected_review_content_digest: verifiedIdentity.reviewContentDigest
  };
}

function normalizeQueueItem(value, applyUiDisabled) {
  const item = record(value);
  const entityKind = requiredString(item.entity_kind, 'item_entity_kind');
  if (!['operator_proposal', 'mixing_agent_action'].includes(entityKind)) {
    throw new Error('decision_queue_invalid_item_entity_kind');
  }
  const queueSection = requiredString(item.queue_section, 'item_queue_section');
  if (!['pending', 'history', 'analysis_only'].includes(queueSection)) {
    throw new Error('decision_queue_invalid_item_queue_section');
  }
  const blockerCodes = Array.isArray(item.apply_blocker_codes)
    ? item.apply_blocker_codes.map((code) => requiredString(code, 'item_apply_blocker_code'))
    : (() => { throw new Error('decision_queue_invalid_item_apply_blocker_codes'); })();
  if (blockerCodes.length > 8) {
    throw new Error('decision_queue_invalid_item_apply_blocker_codes');
  }
  const reviewContractVersion = item.review_contract_version === null
    ? null
    : finiteInteger(item.review_contract_version, 'item_review_contract_version', 1);
  const reviewContentDigest = item.review_content_digest === null
    ? null
    : requiredString(item.review_content_digest, 'item_review_content_digest');
  const serverCanApply = strictBoolean(item.can_apply, 'item_can_apply');
  const detailAvailable = strictBoolean(item.detail_available, 'item_detail_available');
  const identityReady = entityKind === 'operator_proposal'
    && reviewContractVersion === DECISION_QUEUE_CONTRACT_VERSION
    && isSha256Digest(reviewContentDigest)
    && detailAvailable;

  const blockerCount = finiteInteger(
    item.apply_blocker_count,
    'item_apply_blocker_count'
  );
  if (blockerCount < blockerCodes.length) {
    throw new Error('decision_queue_invalid_item_apply_blocker_count');
  }
  return {
    id: requiredString(item.id, 'item_id'),
    entityKind,
    queueSection,
    titlePreview: strictString(item.title_preview, 'item_title_preview'),
    reasonPreview: strictString(item.reason_preview, 'item_reason_preview'),
    displayTarget: strictString(item.display_target, 'item_display_target'),
    kind: requiredString(item.kind, 'item_kind'),
    severity: requiredString(item.severity, 'item_severity'),
    status: requiredString(item.status, 'item_status'),
    source: requiredString(item.source, 'item_source'),
    createdAt: nullableString(item.created_at, 'item_created_at'),
    updatedAt: nullableString(item.updated_at, 'item_updated_at'),
    expiresAt: nullableString(item.expires_at, 'item_expires_at'),
    canApply: serverCanApply && identityReady && !applyUiDisabled,
    requiresApproval: strictBoolean(item.requires_approval, 'item_requires_approval'),
    applyBlockerCodes: blockerCodes,
    applyBlockerCount: blockerCount,
    linkedChangeCount: finiteInteger(item.linked_change_count, 'item_linked_change_count'),
    reviewContractVersion,
    reviewContractStatus: requiredString(
      item.review_contract_status,
      'item_review_contract_status'
    ),
    reviewContentDigest,
    detailAvailable,
    hazardScope: requiredString(item.hazard_scope, 'item_hazard_scope'),
    priorityClass: requiredString(item.priority_class, 'item_priority_class'),
    analysisOnly: strictBoolean(item.analysis_only, 'item_analysis_only'),
    importRequired: strictBoolean(item.import_required, 'item_import_required')
  };
}

export function createUnavailableDecisionQueueState({
  availability = 'unavailable',
  error = '',
  terminalClose = null
} = {}) {
  return {
    contractVersion: DECISION_QUEUE_CONTRACT_VERSION,
    wireSchema: DECISION_QUEUE_WIRE_SCHEMA,
    availability,
    status: 'unavailable',
    snapshot: null,
    page: {
      pageSize: FIRST_PAGE_SIZE,
      returnedCount: 0,
      remainingCount: 0,
      nextCursor: null,
      hasMore: false
    },
    counts: {
      totalCount: 0,
      includedCount: 0,
      omittedCount: 0,
      pinnedTotalCount: 0,
      pinnedIncludedCount: 0,
      pinnedOmittedCount: 0,
      bySection: { pending: 0, history: 0, analysis_only: 0 },
      byPriority: { global_hazard: 0, blocked: 0, high: 0, ordinary: 0, history: 0 },
      byApplyState: { applyable: 0, blocked: 0, analysis_only: 0 }
    },
    items: [],
    overflow: {
      countsOnly: false,
      reason: null,
      unpageableCount: 0,
      pinnedUnpageableCount: 0,
      applyUiDisabled: true,
      hazardDigest: null
    },
    hazards: {
      fatal: false,
      applyUiDisabled: true,
      count: 0,
      codes: [],
      overflowCount: 0,
      sourceCounts: { runtime: 0, manual_write: 0, terminal_result: 0 },
      digest: null
    },
    applyUiDisabled: true,
    paginationLoading: false,
    error,
    terminalClose
  };
}

export function markDecisionQueueLoading(previous, { pagination = false } = {}) {
  if (pagination && previous?.snapshot) {
    return { ...previous, paginationLoading: true, error: '' };
  }
  return {
    ...createUnavailableDecisionQueueState({ availability: 'loading' }),
    paginationLoading: false
  };
}

export function parseDecisionQueueFrame(frame, previous = null) {
  const data = record(frame);
  if (
    data.type !== 'decision_queue'
    || data.contract_version !== DECISION_QUEUE_CONTRACT_VERSION
    || data.wire_schema !== DECISION_QUEUE_WIRE_SCHEMA
    || data.success !== true
  ) {
    throw new Error('decision_queue_v2_contract_required');
  }
  if (!['ok', 'degraded_overflow'].includes(data.status)) {
    throw new Error('decision_queue_invalid_status');
  }
  verifyWireEnvelope(data.wire, 'decision_queue_wire');

  const snapshot = record(data.snapshot);
  const page = record(data.page);
  const counts = record(data.counts);
  const overflow = record(data.overflow);
  const hazards = record(data.hazards);
  const sourceCounts = record(hazards.source_counts);
  const nextCursor = page.next_cursor === null
    ? null
    : requiredString(page.next_cursor, 'page_next_cursor');
  const hazardCodes = Array.isArray(hazards.codes)
    ? hazards.codes.map((code) => requiredString(code, 'hazard_code'))
    : (() => { throw new Error('decision_queue_invalid_hazard_codes'); })();
  if (hazardCodes.length > 16) {
    throw new Error('decision_queue_invalid_hazard_codes');
  }
  const hazardFatal = strictBoolean(hazards.fatal, 'hazards_fatal');
  const hazardApplyDisabled = strictBoolean(
    hazards.apply_ui_disabled,
    'hazards_apply_ui_disabled'
  );
  const overflowApplyDisabled = strictBoolean(
    overflow.apply_ui_disabled,
    'overflow_apply_ui_disabled'
  );
  const applyUiDisabled = hazardFatal || hazardApplyDisabled || overflowApplyDisabled;
  if (!Array.isArray(data.items)) throw new Error('decision_queue_invalid_items');
  const currentItems = data.items.map((item) => normalizeQueueItem(item, applyUiDisabled));
  const snapshotId = requiredString(snapshot.id, 'snapshot_id');
  const snapshotRevision = requiredString(snapshot.revision, 'snapshot_revision');
  const append = previous?.snapshot?.id === snapshotId
    && previous.snapshot.revision === snapshotRevision
    && previous.availability !== 'error'
    && previous.availability !== 'terminal';
  const items = append ? [...previous.items, ...currentItems] : currentItems;
  const returnedCount = finiteInteger(page.returned_count, 'page_returned_count');
  if (returnedCount !== currentItems.length) {
    throw new Error('decision_queue_returned_count_mismatch');
  }
  const hasMore = strictBoolean(page.has_more, 'page_has_more');
  if (hasMore !== Boolean(nextCursor)) {
    throw new Error('decision_queue_cursor_truth_mismatch');
  }

  const normalizedCounts = {
    totalCount: finiteInteger(counts.total_count, 'counts_total_count'),
    includedCount: finiteInteger(counts.included_count, 'counts_included_count'),
    omittedCount: finiteInteger(counts.omitted_count, 'counts_omitted_count'),
    pinnedTotalCount: finiteInteger(counts.pinned_total_count, 'counts_pinned_total_count'),
    pinnedIncludedCount: finiteInteger(
      counts.pinned_included_count,
      'counts_pinned_included_count'
    ),
    pinnedOmittedCount: finiteInteger(
      counts.pinned_omitted_count,
      'counts_pinned_omitted_count'
    ),
    bySection: fixedCountMap(
      counts.by_section,
      ['pending', 'history', 'analysis_only'],
      'counts_by_section'
    ),
    byPriority: fixedCountMap(
      counts.by_priority,
      ['global_hazard', 'blocked', 'high', 'ordinary', 'history'],
      'counts_by_priority'
    ),
    byApplyState: fixedCountMap(
      counts.by_apply_state,
      ['applyable', 'blocked', 'analysis_only'],
      'counts_by_apply_state'
    )
  };
  const pinnedCountTruth = normalizedCounts.pinnedIncludedCount
    + normalizedCounts.pinnedOmittedCount;
  const fixedMapSums = [
    Object.values(normalizedCounts.bySection),
    Object.values(normalizedCounts.byPriority),
    Object.values(normalizedCounts.byApplyState)
  ].map((values) => values.reduce((total, value) => total + value, 0));
  if (
    normalizedCounts.includedCount + normalizedCounts.omittedCount
    !== normalizedCounts.totalCount
    || pinnedCountTruth !== normalizedCounts.pinnedTotalCount
    || normalizedCounts.pinnedTotalCount > normalizedCounts.totalCount
    || fixedMapSums.some((sum) => sum !== normalizedCounts.totalCount)
  ) {
    throw new Error('decision_queue_count_truth_mismatch');
  }
  const remainingCount = finiteInteger(page.remaining_count, 'page_remaining_count');
  const overflowCountsOnly = strictBoolean(overflow.counts_only, 'overflow_counts_only');
  const overflowUnpageableCount = finiteInteger(
    overflow.unpageable_count,
    'overflow_unpageable_count'
  );
  const pinnedUnpageableCount = finiteInteger(
    overflow.pinned_unpageable_count,
    'overflow_pinned_unpageable_count'
  );
  const fullyPageable = strictBoolean(snapshot.fully_pageable, 'snapshot_fully_pageable');
  const hazardCount = finiteInteger(hazards.count, 'hazards_count');
  const hazardOverflowCount = finiteInteger(
    hazards.overflow_count,
    'hazards_overflow_count'
  );
  const normalizedSourceCounts = {
    runtime: finiteInteger(sourceCounts.runtime, 'hazards_source_runtime'),
    manual_write: finiteInteger(sourceCounts.manual_write, 'hazards_source_manual_write'),
    terminal_result: finiteInteger(
      sourceCounts.terminal_result,
      'hazards_source_terminal_result'
    )
  };
  const hazardSourceCount = Object.values(normalizedSourceCounts)
    .reduce((total, value) => total + value, 0);
  const expectedOverflowReason = normalizedCounts.pinnedOmittedCount > 0
    ? 'pinned_metadata_budget'
    : normalizedCounts.omittedCount > 0
      ? 'snapshot_materialization_budget'
      : null;
  const overflowReason = overflow.reason === null
    ? null
    : requiredString(overflow.reason, 'overflow_reason');
  if (
    remainingCount !== normalizedCounts.includedCount - items.length
    || overflowCountsOnly !== (normalizedCounts.omittedCount > 0)
    || overflowUnpageableCount !== normalizedCounts.omittedCount
    || pinnedUnpageableCount !== normalizedCounts.pinnedOmittedCount
    || fullyPageable !== (normalizedCounts.omittedCount === 0)
    || overflowReason !== expectedOverflowReason
    || data.status === 'degraded_overflow' !== (pinnedUnpageableCount > 0)
    || overflowApplyDisabled !== (pinnedUnpageableCount > 0 || hazardApplyDisabled)
    || hazardFatal !== (hazardCount > 0)
    || hazardApplyDisabled !== hazardFatal
    || hazardCount !== hazardCodes.length + hazardOverflowCount
    || hazardSourceCount !== hazardCount
  ) {
    throw new Error('decision_queue_envelope_truth_mismatch');
  }
  const hazardDigest = requiredString(hazards.digest, 'hazards_digest');
  const overflowHazardDigest = requiredString(
    overflow.hazard_digest,
    'overflow_hazard_digest'
  );
  if (hazardDigest !== overflowHazardDigest) {
    throw new Error('decision_queue_hazard_digest_mismatch');
  }

  return {
    contractVersion: DECISION_QUEUE_CONTRACT_VERSION,
    wireSchema: DECISION_QUEUE_WIRE_SCHEMA,
    availability: data.status === 'degraded_overflow' ? 'degraded' : 'ready',
    status: data.status,
    snapshot: {
      id: snapshotId,
      revision: snapshotRevision,
      createdAt: requiredString(snapshot.created_at, 'snapshot_created_at'),
      expiresAt: requiredString(snapshot.expires_at, 'snapshot_expires_at'),
      fullyPageable
    },
    page: {
      pageSize: finiteInteger(page.page_size, 'page_size', 1),
      returnedCount,
      remainingCount,
      nextCursor,
      hasMore
    },
    counts: normalizedCounts,
    items,
    overflow: {
      countsOnly: overflowCountsOnly,
      reason: overflowReason,
      unpageableCount: overflowUnpageableCount,
      pinnedUnpageableCount,
      applyUiDisabled: overflowApplyDisabled,
      hazardDigest: overflowHazardDigest
    },
    hazards: {
      fatal: hazardFatal,
      applyUiDisabled: hazardApplyDisabled,
      count: hazardCount,
      codes: hazardCodes,
      overflowCount: hazardOverflowCount,
      sourceCounts: normalizedSourceCounts,
      digest: hazardDigest
    },
    applyUiDisabled,
    paginationLoading: false,
    error: '',
    terminalClose: null
  };
}

export function proposalReviewIdentityFromQueueItem(item) {
  const source = record(item);
  if (
    source.entityKind !== 'operator_proposal'
    || source.detailAvailable !== true
    || source.reviewContractVersion !== DECISION_QUEUE_CONTRACT_VERSION
    || !isSha256Digest(source.reviewContentDigest)
  ) {
    return null;
  }
  return normalizeReviewIdentity({
    proposalId: source.id,
    reviewContractVersion: source.reviewContractVersion,
    reviewContentDigest: source.reviewContentDigest
  });
}

export function isVerifiedProposalReviewForApply(review, identity) {
  let expected;
  try {
    expected = normalizeReviewIdentity(identity);
  } catch {
    return false;
  }
  const source = record(review);
  const currentState = record(source.currentState);
  const document = record(source.document);
  return source.status === 'verified'
    && source.proposalId === expected.proposalId
    && source.reviewContractVersion === expected.reviewContractVersion
    && source.reviewContentDigest === expected.reviewContentDigest
    && typeof source.verifiedAt === 'string'
    && source.verifiedAt.length > 0
    && currentState.expired === false
    && currentState.can_apply === true
    && currentState.apply_blocker_count === 0
    && Array.isArray(currentState.apply_blocker_codes)
    && currentState.apply_blocker_codes.length === 0
    && document.schema === PROPOSAL_REVIEW_SCHEMA
    && document.proposal_id === expected.proposalId
    && document.review_contract_version === expected.reviewContractVersion;
}

function decodeBase64(value) {
  if (typeof value !== 'string' || value.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(value)) {
    throw new Error('proposal_detail_invalid_base64');
  }
  if (typeof globalThis.atob !== 'function') {
    throw new Error('proposal_detail_base64_decoder_unavailable');
  }
  let binary;
  try {
    binary = globalThis.atob(value);
  } catch {
    throw new Error('proposal_detail_invalid_base64');
  }
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle?.digest) {
    throw new Error('proposal_detail_sha256_unavailable');
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function verifyProposalDetailFrame(
  frame,
  identity,
  expectedChunkIndex,
  expectedMetadata = null
) {
  const expected = normalizeReviewIdentity(identity);
  const data = record(frame);
  const content = record(data.content);
  const currentState = normalizeProposalCurrentState(data.current_state);
  if (
    data.type !== 'operator_proposal_detail'
    || data.contract_version !== DECISION_QUEUE_CONTRACT_VERSION
    || data.status !== 'ok'
    || data.success !== true
  ) {
    throw new Error('operator_proposal_detail_v2_required');
  }
  verifyWireEnvelope(data.wire, 'operator_proposal_detail_wire');
  if (
    data.proposal_id !== expected.proposalId
    || data.review_contract_version !== expected.reviewContractVersion
    || data.review_content_digest !== expected.reviewContentDigest
  ) {
    throw new Error('operator_proposal_detail_identity_mismatch');
  }
  if (
    content.encoding !== 'base64'
    || content.media_type !== PROPOSAL_DETAIL_MEDIA_TYPE
  ) {
    throw new Error('operator_proposal_detail_media_mismatch');
  }
  const chunkIndex = finiteInteger(content.chunk_index, 'detail_chunk_index');
  if (chunkIndex !== expectedChunkIndex) {
    throw new Error('operator_proposal_detail_chunk_order_mismatch');
  }
  const chunkCount = finiteInteger(content.chunk_count, 'detail_chunk_count', 1);
  const totalRawBytes = finiteInteger(content.total_raw_bytes, 'detail_total_raw_bytes');
  const rawChunkBytes = finiteInteger(content.raw_chunk_bytes, 'detail_raw_chunk_bytes');
  if (
    chunkCount > MAX_DETAIL_CHUNKS
    || totalRawBytes > MAX_DETAIL_TOTAL_BYTES
    || rawChunkBytes > MAX_DETAIL_CHUNK_BYTES
    || chunkIndex >= chunkCount
  ) {
    throw new Error('operator_proposal_detail_size_invariant_breached');
  }
  if (
    typeof content.chunk_sha256 !== 'string'
    || !SHA256_HEX_PATTERN.test(content.chunk_sha256)
    || typeof content.content_sha256 !== 'string'
    || !SHA256_HEX_PATTERN.test(content.content_sha256)
    || `sha256:${content.content_sha256}` !== expected.reviewContentDigest
  ) {
    throw new Error('operator_proposal_detail_digest_mismatch');
  }
  const currentStateKey = JSON.stringify(currentState);
  if (expectedMetadata && (
    expectedMetadata.chunkCount !== chunkCount
    || expectedMetadata.totalRawBytes !== totalRawBytes
    || expectedMetadata.contentSha256 !== content.content_sha256
    || expectedMetadata.currentStateKey !== currentStateKey
  )) {
    throw new Error('operator_proposal_detail_chunk_metadata_changed');
  }
  const bytes = decodeBase64(content.data);
  if (bytes.byteLength !== rawChunkBytes) {
    throw new Error('operator_proposal_detail_chunk_size_mismatch');
  }
  if (await sha256Hex(bytes) !== content.chunk_sha256) {
    throw new Error('operator_proposal_detail_chunk_digest_mismatch');
  }
  return {
    bytes,
    chunkIndex,
    chunkCount,
    totalRawBytes,
    contentSha256: content.content_sha256,
    currentState,
    currentStateKey
  };
}

export async function verifyProposalDetailFrames(frames, identity) {
  const expected = normalizeReviewIdentity(identity);
  if (!Array.isArray(frames) || frames.length === 0) {
    throw new Error('operator_proposal_detail_frames_required');
  }
  const verifiedFrames = [];
  let metadata = null;
  for (let index = 0; index < frames.length; index += 1) {
    const verified = await verifyProposalDetailFrame(frames[index], expected, index, metadata);
    metadata ??= verified;
    verifiedFrames.push(verified);
  }
  if (metadata.chunkCount !== verifiedFrames.length) {
    throw new Error('operator_proposal_detail_incomplete');
  }
  const bytes = new Uint8Array(metadata.totalRawBytes);
  let offset = 0;
  for (const frame of verifiedFrames) {
    if (offset + frame.bytes.byteLength > bytes.byteLength) {
      throw new Error('operator_proposal_detail_total_size_mismatch');
    }
    bytes.set(frame.bytes, offset);
    offset += frame.bytes.byteLength;
  }
  if (offset !== bytes.byteLength) {
    throw new Error('operator_proposal_detail_total_size_mismatch');
  }
  const digest = await sha256Hex(bytes);
  if (digest !== metadata.contentSha256 || `sha256:${digest}` !== expected.reviewContentDigest) {
    throw new Error('operator_proposal_detail_content_digest_mismatch');
  }
  let document;
  try {
    document = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch {
    throw new Error('operator_proposal_detail_content_invalid_json');
  }
  const review = record(document);
  if (
    review.schema !== PROPOSAL_REVIEW_SCHEMA
    || review.proposal_id !== expected.proposalId
    || review.review_contract_version !== expected.reviewContractVersion
  ) {
    throw new Error('operator_proposal_detail_document_identity_mismatch');
  }
  return {
    status: 'verified',
    proposalId: expected.proposalId,
    reviewContractVersion: expected.reviewContractVersion,
    reviewContentDigest: expected.reviewContentDigest,
    chunkCount: metadata.chunkCount,
    totalRawBytes: metadata.totalRawBytes,
    currentState: metadata.currentState,
    document,
    verifiedAt: new Date().toISOString()
  };
}
