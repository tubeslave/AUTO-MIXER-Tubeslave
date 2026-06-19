import type { DirectorDecisionMock, DirectorEventMock } from '../../mock/dashboardMock';
import type { DirectorRuntime, Task, Tone } from '../../types';

export interface DirectorDecisionRow {
  time: string;
  title: string;
  target: string;
  status: string;
  tone: Tone;
}

export interface DirectorProposalRow {
  id: string;
  agent: string;
  title: string;
  detail: string;
  status: string;
  tone: Tone;
  reviewMessage: string;
  rejectMessage: string;
}

export type DirectorEventRow = DirectorEventMock;

export function isBackendDirectorSource(sourceLabel: string): boolean {
  return sourceLabel.toLowerCase().includes('backend');
}

export function directorFallbackRowsForSource<T>(sourceLabel: string, fallbackRows: T[]): T[] {
  return isBackendDirectorSource(sourceLabel) ? [] : fallbackRows;
}

export function toneForDirectorStatus(status: string, priority?: Task['priority']): Tone {
  const normalized = status.toLowerCase();
  if (normalized === 'applied') return 'green';
  if (normalized === 'approved') return 'green';
  if (normalized === 'safe') return 'green';
  if (normalized.includes('rejected')) return 'red';
  if (normalized.includes('blocked')) return 'red';
  if (normalized.includes('failed')) return 'red';
  if (normalized.includes('review')) return 'yellow';
  if (normalized === 'pending') return 'yellow';
  if (normalized === 'queued') return 'yellow';
  if (priority === 'High') return 'yellow';
  return 'blue';
}

export function taskRequiresReviewOnly(task: Task): boolean {
  const backendStatus = (task.backendStatus ?? '').toLowerCase();
  return Boolean(
    task.reviewOnly
    || task.canApply === false
    || (task.applyBlockers?.length ?? 0) > 0
    || backendStatus.includes('proposal_only')
    || backendStatus.includes('blocked')
    || backendStatus.includes('unsupported')
  );
}

export function directorDecisionRows(
  taskRows: Task[],
  fallbackDecisions: DirectorDecisionMock[],
  sourceLabel = 'Mock'
): DirectorDecisionRow[] {
  if (!taskRows.length) {
    return directorFallbackRowsForSource(sourceLabel, fallbackDecisions);
  }

  return taskRows.slice(0, 4).map(task => ({
    time: task.time,
    title: task.title,
    target: task.target ?? task.id,
    status: task.status,
    tone: toneForDirectorStatus(task.status, task.priority)
  }));
}

function detailForProposalTask(task: Task): string {
  if (task.reason) {
    return task.reason;
  }

  if (task.applyBlockers?.length) {
    return `${task.applyBlockers.slice(0, 2).join(', ')} - review only`;
  }

  const target = task.target ? `${task.target} - ` : '';
  const valueType = task.requestedValueType ? `${task.requestedValueType} proposal` : task.backendStatus ?? task.priority;
  const policy = taskRequiresReviewOnly(task) ? 'review only' : 'supervised gate';
  return `${target}${valueType} - ${policy}`;
}

function statusForProposalTask(task: Task): string {
  if (taskRequiresReviewOnly(task)) {
    return 'Review only';
  }
  return task.status;
}

export function directorProposalRows(
  taskRows: Task[],
  fallbackRows: DirectorProposalRow[],
  sourceLabel = 'Mock'
): DirectorProposalRow[] {
  if (!taskRows.length) {
    return directorFallbackRowsForSource(sourceLabel, fallbackRows);
  }

  return taskRows
    .filter(task => task.status === 'Pending' || task.status === 'Review')
    .slice(0, 2)
    .map(task => ({
      id: task.id,
      agent: task.agent,
      title: task.title,
      detail: detailForProposalTask(task),
      status: statusForProposalTask(task),
      tone: taskRequiresReviewOnly(task)
        ? 'red'
        : toneForDirectorStatus(task.status, task.priority),
      reviewMessage: `${task.id} reviewed locally. No OSC write sent.`,
      rejectMessage: `${task.id} rejected locally. No OSC write sent.`
    }));
}

function appendRuntimeEvent(
  rows: DirectorEventRow[],
  source: string,
  message: string,
  tone: Tone
) {
  if (!message.trim()) {
    return;
  }

  rows.push({
    time: 'Now',
    source,
    message,
    tone
  });
}

function integrationBundleEventMessage(blockedReasons: string[]): string {
  const bundleReasons = blockedReasons.filter(reason => reason.startsWith('integration_bundle_'));
  if (!bundleReasons.length) {
    return '';
  }

  const status = bundleReasons
    .find(reason => reason.startsWith('integration_bundle_status:'))
    ?.split(':')
    .slice(1)
    .join(':') || 'blocked';
  const blockerCount = bundleReasons.filter(reason => reason.startsWith('integration_bundle_blocker:')).length;
  const missingChecks = bundleReasons
    .filter(reason => reason.startsWith('integration_bundle_missing_check:'))
    .map(reason => reason.split(':').slice(1).join(':'))
    .filter(Boolean);
  const sourceAuthorityDenied = bundleReasons.includes('integration_bundle_source_authority_denied:true');
  const liveAuthorityDenied = bundleReasons.includes('integration_bundle_live_authority_granted:false');
  const readinessClosureDenied = bundleReasons.includes('integration_bundle_readiness_closure_allowed:false');
  const authorityDetails = [
    sourceAuthorityDenied ? 'source authority denied' : '',
    liveAuthorityDenied ? 'live authority denied' : '',
    readinessClosureDenied ? 'readiness closure denied' : ''
  ].filter(Boolean);
  const detail = [
    blockerCount ? `${blockerCount} blocker(s)` : '',
    missingChecks.length ? `missing: ${missingChecks.slice(0, 2).join(', ')}` : '',
    authorityDetails.join(', ')
  ].filter(Boolean).join('; ');

  return detail
    ? `Integration bundle ${status}; ${detail}.`
    : `Integration bundle ${status}.`;
}

export function directorEventRows(
  runtime: DirectorRuntime,
  fallbackRows: DirectorEventMock[]
): DirectorEventRow[] {
  if (!isBackendDirectorSource(runtime.sourceLabel)) {
    return fallbackRows;
  }

  const rows: DirectorEventRow[] = [];
  appendRuntimeEvent(
    rows,
    'Safety',
    `Live write policy: ${runtime.liveWritePolicy}; manual gate: ${runtime.manualGateLabel}.`,
    runtime.manualGateTone
  );
  appendRuntimeEvent(
    rows,
    'Readiness',
    `Ready: ${runtime.readyLabel}; audio: ${runtime.audioLabel}; proposals: ${runtime.proposalQueueLabel}.`,
    runtime.readyTone
  );

  if (runtime.resetRequired) {
    appendRuntimeEvent(
      rows,
      'Safety',
      'Emergency reset is required before any supervised apply path can continue.',
      'red'
    );
  }

  const integrationBundleMessage = integrationBundleEventMessage(runtime.blockedReasons);
  if (integrationBundleMessage) {
    appendRuntimeEvent(rows, 'Integration bundle', integrationBundleMessage, 'red');
  }

  const nonIntegrationBlockers = runtime.blockedReasons.filter(
    reason => !reason.startsWith('integration_bundle_')
  );
  if (nonIntegrationBlockers.length || (runtime.blockedReasons.length && !integrationBundleMessage)) {
    appendRuntimeEvent(
      rows,
      'Blockers',
      (nonIntegrationBlockers.length ? nonIntegrationBlockers : runtime.blockedReasons)
        .slice(0, 2)
        .join(', '),
      'red'
    );
  } else if (runtime.hardwareProofItems.length) {
    const proofSummary = runtime.hardwareProofItems
      .slice(0, 2)
      .map(item => `${item.label}: ${item.value}`)
      .join('; ');
    appendRuntimeEvent(rows, 'Proof', proofSummary, runtime.hardwareProofItems[0].tone);
  }

  return rows;
}
