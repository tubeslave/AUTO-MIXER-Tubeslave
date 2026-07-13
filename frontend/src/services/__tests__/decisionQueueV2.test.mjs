import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildDecisionQueueFirstPageRequest,
  buildDecisionQueueNextPageRequest,
  buildProposalApplyRequest,
  buildProposalDetailRequest,
  createUnavailableDecisionQueueState,
  isVerifiedProposalReviewForApply,
  parseDecisionQueueFrame,
  proposalReviewIdentityFromQueueItem,
  verifyProposalDetailFrames
} from '../decisionQueueV2.mjs';

const EMPTY_DIGEST = `sha256:${'0'.repeat(64)}`;

function queueItem(id, overrides = {}) {
  return {
    id,
    entity_kind: 'operator_proposal',
    queue_section: 'pending',
    title_preview: `Proposal ${id}`,
    reason_preview: 'Bounded reason',
    display_target: 'CH 01',
    kind: 'fader',
    severity: 'high',
    status: 'pending',
    source: 'operator_proposal_queue',
    created_at: '2026-07-13T12:00:00Z',
    updated_at: '2026-07-13T12:00:00Z',
    expires_at: '2026-07-13T12:10:00Z',
    can_apply: true,
    requires_approval: true,
    apply_blocker_codes: [],
    apply_blocker_count: 0,
    linked_change_count: 0,
    review_contract_version: 2,
    review_contract_status: 'bound',
    review_content_digest: EMPTY_DIGEST,
    detail_available: true,
    hazard_scope: 'proposal_local',
    priority_class: 'high',
    analysis_only: false,
    import_required: false,
    ...overrides
  };
}

function queueFrame({
  snapshotId = 'snapshot-1',
  items = [],
  nextCursor = null,
  remainingCount = 0,
  totalCount = items.length,
  omittedCount = 0,
  status = 'ok',
  hazardCodes = [],
  overflowApplyDisabled = false,
  pinnedOmittedCount = 0
} = {}) {
  const fatal = hazardCodes.length > 0;
  return {
    type: 'decision_queue',
    contract_version: 2,
    wire_schema: 'automixer.decision_queue.v2',
    status,
    success: true,
    snapshot: {
      id: snapshotId,
      revision: EMPTY_DIGEST,
      created_at: '2026-07-13T12:00:00Z',
      expires_at: '2026-07-13T12:00:30Z',
      fully_pageable: omittedCount === 0
    },
    page: {
      page_size: 50,
      returned_count: items.length,
      remaining_count: remainingCount,
      next_cursor: nextCursor,
      has_more: nextCursor !== null
    },
    counts: {
      total_count: totalCount,
      included_count: totalCount - omittedCount,
      omitted_count: omittedCount,
      pinned_total_count: pinnedOmittedCount,
      pinned_included_count: 0,
      pinned_omitted_count: pinnedOmittedCount,
      by_section: { pending: totalCount, history: 0, analysis_only: 0 },
      by_priority: { global_hazard: 0, blocked: 0, high: totalCount, ordinary: 0, history: 0 },
      by_apply_state: { applyable: totalCount, blocked: 0, analysis_only: 0 }
    },
    items,
    overflow: {
      counts_only: omittedCount > 0,
      reason: pinnedOmittedCount > 0
        ? 'pinned_metadata_budget'
        : omittedCount > 0
          ? 'snapshot_materialization_budget'
          : null,
      unpageable_count: omittedCount,
      pinned_unpageable_count: pinnedOmittedCount,
      apply_ui_disabled: overflowApplyDisabled || fatal,
      hazard_digest: EMPTY_DIGEST
    },
    hazards: {
      fatal,
      apply_ui_disabled: fatal,
      count: hazardCodes.length,
      codes: hazardCodes,
      overflow_count: 0,
      source_counts: {
        runtime: hazardCodes.length,
        manual_write: 0,
        terminal_result: 0
      },
      digest: EMPTY_DIGEST
    },
    wire: { max_bytes: 131072, serialized_bytes: 1000 }
  };
}

test('builds the exact first and later Decision Queue v2 request frames', () => {
  assert.deepEqual(buildDecisionQueueFirstPageRequest(), {
    type: 'get_decision_queue',
    contract_version: 2,
    page_size: 50
  });
  assert.deepEqual(buildDecisionQueueNextPageRequest('cursor-2', EMPTY_DIGEST), {
    type: 'get_decision_queue',
    contract_version: 2,
    cursor: 'cursor-2',
    expected_snapshot_revision: EMPTY_DIGEST
  });
  assert.equal('page_size' in buildDecisionQueueNextPageRequest('cursor-2', EMPTY_DIGEST), false);
});

test('preserves server order across pages and treats a real empty items frame as empty', () => {
  const first = parseDecisionQueueFrame(queueFrame({
    items: [queueItem('P-2'), queueItem('P-1')],
    nextCursor: 'cursor-2',
    remainingCount: 1,
    totalCount: 3
  }));
  const second = parseDecisionQueueFrame(queueFrame({
    items: [queueItem('P-3')],
    totalCount: 3
  }), first);

  assert.deepEqual(second.items.map(item => item.id), ['P-2', 'P-1', 'P-3']);
  assert.deepEqual(parseDecisionQueueFrame(queueFrame()).items, []);
  assert.deepEqual(createUnavailableDecisionQueueState().items, []);
});

test('fatal hazards and pinned overflow keep every compact row non-applyable', () => {
  const hazard = parseDecisionQueueFrame(queueFrame({
    items: [queueItem('P-hazard')],
    hazardCodes: ['emergency_stop_active']
  }));
  assert.equal(hazard.applyUiDisabled, true);
  assert.equal(hazard.items[0].canApply, false);

  const overflow = parseDecisionQueueFrame(queueFrame({
    items: [queueItem('P-overflow')],
    totalCount: 2,
    omittedCount: 1,
    status: 'degraded_overflow',
    overflowApplyDisabled: true,
    pinnedOmittedCount: 1
  }));
  assert.equal(overflow.overflow.countsOnly, true);
  assert.equal(overflow.overflow.unpageableCount, 1);
  assert.equal(overflow.items[0].canApply, false);
});

test('detail and apply requests carry the exact immutable review identity', () => {
  const parsed = parseDecisionQueueFrame(queueFrame({ items: [queueItem('P-7')] }));
  const identity = proposalReviewIdentityFromQueueItem(parsed.items[0]);
  assert.equal(isVerifiedProposalReviewForApply({
    status: 'idle',
    proposalId: 'P-7',
    reviewContractVersion: 2,
    reviewContentDigest: EMPTY_DIGEST
  }, identity), false);
  assert.deepEqual(buildProposalDetailRequest(identity, 0), {
    type: 'get_operator_proposal_detail',
    contract_version: 2,
    proposal_id: 'P-7',
    expected_review_contract_version: 2,
    expected_review_content_digest: EMPTY_DIGEST,
    chunk_index: 0
  });
  assert.deepEqual(buildProposalApplyRequest({
    identity,
    approved: true,
    reason: 'frontend_verified_detail_apply',
    approvalId: 'frontend-P-7',
    idempotencyKey: 'frontend-P-7-1'
  }), {
    type: 'apply_operator_proposal',
    proposal_id: 'P-7',
    approved: true,
    reason: 'frontend_verified_detail_apply',
    approval_id: 'frontend-P-7',
    idempotency_key: 'frontend-P-7-1',
    expected_review_contract_version: 2,
    expected_review_content_digest: EMPTY_DIGEST
  });
  assert.throws(() => buildProposalApplyRequest({
    identity: { ...identity, reviewContentDigest: '' },
    approved: true,
    reason: 'x',
    approvalId: 'x',
    idempotencyKey: 'x'
  }), /digest_unverified/);
});

async function sha256Hex(bytes) {
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Buffer.from(digest).toString('hex');
}

async function detailFrames(document, splitAt) {
  const bytes = new TextEncoder().encode(JSON.stringify(document));
  const contentSha256 = await sha256Hex(bytes);
  const chunks = [bytes.slice(0, splitAt), bytes.slice(splitAt)];
  const digest = `sha256:${contentSha256}`;
  const frames = [];
  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    frames.push({
      type: 'operator_proposal_detail',
      contract_version: 2,
      status: 'ok',
      success: true,
      proposal_id: document.proposal_id,
      review_contract_version: 2,
      review_content_digest: digest,
      current_state: {
        status: 'pending',
        updated_at: '2026-07-13T12:00:00Z',
        expired: false,
        can_apply: true,
        apply_blocker_codes: [],
        apply_blocker_count: 0
      },
      content: {
        encoding: 'base64',
        media_type: 'application/vnd.automixer.operator-proposal-review+json;version=2',
        chunk_index: index,
        chunk_count: chunks.length,
        raw_chunk_bytes: chunk.byteLength,
        total_raw_bytes: bytes.byteLength,
        chunk_sha256: await sha256Hex(chunk),
        content_sha256: contentSha256,
        data: Buffer.from(chunk).toString('base64')
      },
      wire: { max_bytes: 131072, serialized_bytes: 1000 }
    });
  }
  return { digest, frames };
}

test('verifies every detail chunk and the final canonical content before returning verified', async () => {
  const document = {
    schema: 'automixer.operator_proposal.review_content.v2',
    proposal_id: 'P-detail',
    review_contract_version: 2,
    title: 'Unicode detail — вокал',
    raw: { transaction_plan: [{ kind: 'fader', value: -4.5 }] }
  };
  const { digest, frames } = await detailFrames(document, 45);
  const verified = await verifyProposalDetailFrames(frames, {
    proposalId: 'P-detail',
    reviewContractVersion: 2,
    reviewContentDigest: digest
  });

  assert.equal(verified.status, 'verified');
  assert.equal(verified.chunkCount, 2);
  assert.deepEqual(verified.document, document);
  assert.equal(isVerifiedProposalReviewForApply(verified, {
    proposalId: 'P-detail',
    reviewContractVersion: 2,
    reviewContentDigest: digest
  }), true);
  assert.equal(isVerifiedProposalReviewForApply({
    ...verified,
    reviewContentDigest: `sha256:${'f'.repeat(64)}`
  }, {
    proposalId: 'P-detail',
    reviewContractVersion: 2,
    reviewContentDigest: digest
  }), false);

  const tampered = structuredClone(frames);
  tampered[1].content.data = Buffer.from('tampered').toString('base64');
  await assert.rejects(
    verifyProposalDetailFrames(tampered, {
      proposalId: 'P-detail',
      reviewContractVersion: 2,
      reviewContentDigest: digest
    }),
    /chunk_size_mismatch|chunk_digest_mismatch/
  );

  const malformedState = structuredClone(frames);
  delete malformedState[0].current_state.apply_blocker_count;
  await assert.rejects(
    verifyProposalDetailFrames(malformedState, {
      proposalId: 'P-detail',
      reviewContractVersion: 2,
      reviewContentDigest: digest
    }),
    /current_state_apply_blocker_count/
  );
});
