import {
  ActionCluster,
  DataTable,
  MetricCard,
  Panel,
  StatusBadge
} from './AutomixerComponents';
import { directorBlockedReasonMessages } from '../dashboard/directorPanelModel';
import { applyPlanForTask, ensembleEvidenceForTask, safetyEvidenceForTask } from '../../mock/automixerData';
import { getModeDisplayLabel, textClassForTone, toneForMode, toneForPriority, toneFromState } from '../../utils/prototypeTone';
import type {
  ApplyPlanStep,
  ChannelIntelligenceRecord,
  DecisionTraceRecord,
  DecisionPolicy,
  DirectorRuntime,
  Metric,
  GoalCapabilityObjectiveReadinessByGoalArea,
  ProposalPreview,
  ReplayRunRecord,
  SafetyMonitorState,
  SafetyEvidenceItem,
  Task,
  GoalCapabilityObjectiveReadinessHotspot,
  GoalCapabilityObjectiveReadinessPriority,
  Tone,
  UiActions,
  UiState
} from '../../types';

export type TaskRow = Task & {
  proposal: ProposalPreview;
};

function taskQueueMetrics(
  taskRows: TaskRow[],
  integrationBundleApplyBlocked = false,
): Metric[] {
  const applyableCount = integrationBundleApplyBlocked
    ? 0
    : taskRows.filter(task => task.canApply !== false).length;
  const reviewOnlyCount = integrationBundleApplyBlocked
    ? taskRows.length
    : taskRows.filter(task => task.reviewOnly || task.canApply === false).length;

  return [
    { label: 'Total Queue', value: String(taskRows.length), sub: 'operator proposals', tone: 'blue' },
    {
      label: integrationBundleApplyBlocked ? 'Apply Blocked' : 'Applyable',
      value: String(applyableCount),
      sub: integrationBundleApplyBlocked ? 'integration bundle' : 'fader / gain only',
      tone: integrationBundleApplyBlocked ? 'red' : 'green'
    },
    { label: 'Review Only', value: String(reviewOnlyCount), sub: 'no live apply', tone: 'red' },
    { label: 'Pending', value: String(taskRows.filter(task => task.status === 'Pending').length), sub: 'operator wait', tone: 'yellow' },
    { label: 'Applied', value: String(taskRows.filter(task => task.status === 'Applied').length), sub: 'supervised history', tone: 'green' }
  ];
}

function goalAreaReadinessSummary(rows: GoalCapabilityObjectiveReadinessPriority[]): string {
  const row = rows[0];
  if (!row) return 'none';
  const goalArea = row.goal_area || 'unknown';
  const readiness = typeof row.readiness_score === 'number'
    ? `${Math.round(row.readiness_score)}%`
    : 'n/a';
  return `${goalArea}: ${readiness}`;
}

function hotspotSummary(rows: GoalCapabilityObjectiveReadinessHotspot[]): string {
  if (!rows.length) return 'none';
  const row = rows[0];
  const area = row.goal_area || 'unknown';
  const gaps = row.gap_row_count ?? 0;
  const blockers = row.blocker_row_ids ?? [];
  const blockersSummary = blockers.length ? blockers.slice(0, 3).join(', ') : 'none';
  return `${area}: ${gaps} gap(s), blockers ${blockersSummary}`;
}

function topBlockedGoalAreaRows(
  rows: Record<string, GoalCapabilityObjectiveReadinessByGoalArea>,
): GoalCapabilityObjectiveReadinessByGoalArea[] {
  return Object.entries(rows)
    .map(([goalArea, data]) => ({
      ...data,
      goal_area: data.goal_area || goalArea,
    }))
    .sort((left, right) => {
      const gapDelta = (right.gap_row_count ?? 0) - (left.gap_row_count ?? 0);
      if (gapDelta !== 0) return gapDelta;
      const blockerDelta = (right.blocker_row_count ?? 0) - (left.blocker_row_count ?? 0);
      if (blockerDelta !== 0) return blockerDelta;
      return left.goal_area.localeCompare(right.goal_area);
    });
}

function blockedAreaSummary(rows: Record<string, GoalCapabilityObjectiveReadinessByGoalArea>): string {
  const top = topBlockedGoalAreaRows(rows)[0];
  if (!top) return 'none';
  const goalArea = top.goal_area || 'unknown';
  const gaps = top.gap_row_count ?? 0;
  const blockers = top.blocker_row_ids ?? [];
  const blockerLabel = blockers.length ? blockers.join(', ') : 'none';
  return `${goalArea}: ${gaps} gap(s), blockers ${blockerLabel}`;
}

function toneForReadinessScore(score: number): Tone {
  if (score >= 80) return 'green';
  if (score >= 40) return 'yellow';
  return 'red';
}

function focusTaskDetail(actions: UiActions, task: TaskRow) {
  actions.selectTask(task.id);
  actions.focusAction({
    action: 'Detail',
    target: task.title,
    detail: `Review task ${task.id} from ${task.agent}. This only changes the local inspector context.`,
    tone: toneForPriority(task.priority),
    timestamp: '10:24:15'
  });
}

function requestTaskApply(
  task: TaskRow,
  ui: UiState,
  actions: UiActions,
  integrationBundleApplyBlocked = false
) {
  if (integrationBundleApplyBlocked) {
    actions.selectTask(task.id);
    actions.focusAction({
      action: 'Apply blocked',
      target: task.title,
      detail: 'Apply is blocked: offline integration bundle blockers are visible. Supervised apply remains review-only.',
      tone: 'red',
      timestamp: '10:24:15'
    });
    actions.pushToast(`${task.title} apply remains blocked by integration bundle evidence.`, 'red');
    return;
  }

  if (task.canApply === false || task.reviewOnly) {
    actions.selectTask(task.id);
    actions.focusAction({
      action: 'Review only',
      target: task.title,
      detail: `Apply is blocked: ${(task.applyBlockers ?? ['unsupported_value_type']).join(', ')}.`,
      tone: 'red',
      timestamp: '10:24:15'
    });
    actions.pushToast(`${task.title} is review-only. Live apply is blocked.`, 'red');
    return;
  }

  if (ui.mode === 'Supervised') {
    actions.openSupervisedProposal(task.id);
    return;
  }

  actions.setTaskStatus(task.id, 'Applied');
}

export function TaskDetailsPanel({
  task,
  ui,
  actions,
  integrationBundleApplyBlocked = false,
  policy
}: {
  task: TaskRow;
  ui: UiState;
  actions: UiActions;
  integrationBundleApplyBlocked?: boolean;
  policy: DecisionPolicy;
}) {
  const proposal = task.proposal;
  const applyBlocked = integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly;
  const blockers = task.applyBlockers ?? [];
  const detailApplyBlockers = integrationBundleApplyBlocked
    ? Array.from(new Set(['offline_integration_bundle_blockers_visible', ...blockers]))
    : blockers;
  const ensembleEvidence = ensembleEvidenceForTask(task);

  return (
    <Panel title="Task Details">
      <div className="mb-3 flex items-center justify-between gap-2">
        <StatusBadge tone={toneForPriority(task.priority)}>{task.priority}</StatusBadge>
        <span className="font-mono text-[10px] text-muted">{task.id}</span>
      </div>
      <h3 className="text-base font-bold">{task.title}</h3>
      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px]">
        <StatusBadge tone={toneFromState(task.status)}>{task.status}</StatusBadge>
        <StatusBadge tone={toneForMode(ui.mode)}>{getModeDisplayLabel(ui.mode)}</StatusBadge>
        {(integrationBundleApplyBlocked || task.reviewOnly) && (
          <StatusBadge tone="red">{integrationBundleApplyBlocked ? 'Blocked' : 'Review only'}</StatusBadge>
        )}
        <span className="rounded-[4px] border border-line bg-row px-1.5 py-0.5 text-muted">{task.agent}</span>
        <span className="font-mono text-muted">{task.time}</span>
      </div>
      <div className="mt-3 rounded-[6px] border border-line bg-row p-2">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="am-label">Proposal Preview</div>
          <StatusBadge tone={proposal.tone}>{proposal.risk}</StatusBadge>
        </div>
        <div className="grid grid-cols-[76px_1fr] gap-y-1.5 text-[10px]">
          <span className="text-muted">Target</span><span className="truncate text-text">{task.target ?? 'Session'}</span>
          <span className="text-muted">Parameter</span><span className="truncate text-text">{proposal.parameter}</span>
          <span className="text-muted">Delta</span><span className="truncate font-bold text-audio">{proposal.delta}</span>
          <span className="text-muted">Proof</span><span className="truncate text-text">{proposal.proof}</span>
          <span className="text-muted">Rollback</span><span className="truncate text-text">{proposal.rollback}</span>
          {task.requestedValueType && (
            <>
              <span className="text-muted">Value type</span><span className="truncate text-danger">{task.requestedValueType.replace(/_/g, ' ')}</span>
            </>
          )}
        </div>
      </div>
      {task.reason && (
        <div className="mt-3 border-l-2 border-audio/40 pl-2">
          <div className="am-label mb-1">Backend Reason</div>
          <p className="break-words text-[10px] leading-4 text-muted">{task.reason}</p>
        </div>
      )}
      {ensembleEvidence.length > 0 && (
        <div className="mt-3 rounded-[6px] border border-warn/30 bg-warn/10 p-2">
          <div className="am-label mb-2 text-warn">SafeGain Ensemble Context</div>
          <div className="grid grid-cols-2 gap-1">
            {ensembleEvidence.map(row => (
              <div className="min-w-0 rounded-[5px] border border-line bg-panel2 px-1.5 py-1" key={`${row.label}-${row.value}`}>
                <div className="truncate text-[8px] uppercase tracking-[0.06em] text-dim">{row.label}</div>
                <div className={`truncate text-[9px] font-bold ${textClassForTone(row.tone)}`}>{row.value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {detailApplyBlockers.length > 0 && (
        <div className="mt-3 rounded-[6px] border border-danger/30 bg-danger/10 p-2">
          <div className="am-label mb-1 text-danger">Apply Blockers</div>
          <div className="space-y-1 text-[10px] text-muted">
            {detailApplyBlockers.slice(0, 4).map(blocker => (
              <div className="truncate" key={blocker}>{blocker.replace(/_/g, ' ')}</div>
            ))}
          </div>
        </div>
      )}
      <div className="mt-3">
        <div className="am-label mb-2">Decision Chain</div>
        <div className="space-y-1.5">
          {policy.chain.map((step, index) => (
            <div className="grid grid-cols-[16px_1fr_68px] items-center gap-2 text-[10px]" key={`${step.node}-${step.state}`}>
              <span className="grid h-4 w-4 place-items-center rounded-full border border-line bg-panel2 font-mono text-[8px] text-muted">{index + 1}</span>
              <span className="truncate text-muted">{step.node}</span>
              <StatusBadge tone={step.tone}>{step.state}</StatusBadge>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[10px] leading-4 text-muted">{policy.hint}</p>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2">
        {applyBlocked ? (
          <button className="am-button text-audio" onClick={() => focusTaskDetail(actions, task)} type="button">Inspect</button>
        ) : (
          <button className="am-button text-online" onClick={() => requestTaskApply(task, ui, actions, integrationBundleApplyBlocked)} type="button">{policy.actionLabel}</button>
        )}
        <button className="am-button text-danger" onClick={() => actions.setTaskStatus(task.id, 'Rejected')} type="button">Reject</button>
      </div>
    </Panel>
  );
}

function ApplyPlanPanel({ plan }: { plan: ApplyPlanStep[] }) {
  return (
    <Panel title="Apply Plan">
      <div className="grid grid-cols-5 gap-1">
        {plan.map(step => (
          <div className="rounded-[5px] border border-line bg-row px-1.5 py-1" key={`${step.label}-${step.value}`}>
            <div className="truncate text-[8px] uppercase tracking-[0.04em] text-dim">{step.label}</div>
            <div className={`mt-1 truncate text-[9px] font-bold ${textClassForTone(step.tone)}`}>{step.value}</div>
          </div>
        ))}
      </div>
      <div className="mt-2 h-1 overflow-hidden rounded-full bg-line">
        <div className="h-full w-3/5 rounded-full bg-gradient-to-r from-audio via-warn to-online" />
      </div>
    </Panel>
  );
}

function SafetyEvidencePanel({ evidence }: { evidence: SafetyEvidenceItem[] }) {
  return (
    <Panel title="Safety Evidence">
      <div className="grid grid-cols-3 gap-1">
        {evidence.map(row => (
          <div className="rounded-[5px] border border-line bg-row px-1.5 py-1" key={`${row.label}-${row.value}`}>
            <div className="truncate text-[8px] uppercase tracking-[0.08em] text-dim">{row.label}</div>
            <div className={`truncate text-[9px] font-bold ${textClassForTone(row.tone)}`}>{row.value}</div>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function OperatorDecisionPanel({ ui, policy }: { ui: UiState; policy: DecisionPolicy }) {
  return (
    <Panel title="Operator Decision">
      <div className="space-y-2">
        {[
          { label: 'Current mode', value: getModeDisplayLabel(ui.mode), tone: toneForMode(ui.mode) },
          { label: 'Apply result', value: policy.apply, tone: toneForMode(ui.mode) },
          { label: 'Write path', value: ui.backend.applyPath, tone: ui.backend.canApplyToConsole ? 'yellow' as const : 'red' as const },
          { label: 'Backend', value: ui.backend.statusLabel, tone: ui.backend.statusTone }
        ].map(row => (
          <div className="flex items-center justify-between border-b border-line pb-2 text-[10px]" key={`${row.label}-${row.value}`}>
            <span className="text-muted">{row.label}</span>
            <StatusBadge tone={row.tone}>{row.value}</StatusBadge>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function SafetyMonitorPanel({ safety }: { safety: SafetyMonitorState }) {
  const readinessPriority = goalAreaReadinessSummary(safety.goalCapabilityObjectiveReadinessGoalAreaPriorities);
  const proofReadinessPriority = goalAreaReadinessSummary(safety.proofGoalCapabilityObjectiveReadinessGoalAreaPriorities);
  const hotspotText = hotspotSummary(safety.goalCapabilityObjectiveReadinessHotspots);
  const proofHotspotText = hotspotSummary(safety.proofGoalCapabilityObjectiveReadinessHotspots);
  const proofBlockedAreaText = blockedAreaSummary(safety.proofGoalCapabilityObjectiveReadinessByGoalArea);

  return (
    <Panel title="Safety Monitor">
      <div className="grid grid-cols-2 gap-1">
        {[
          { label: 'Mode', value: safety.operatorMode, tone: safety.operatorMode === 'supervised' ? 'yellow' as const : 'blue' as const },
          { label: 'Policy', value: safety.liveWritePolicy, tone: safety.liveWritePolicy.includes('fader_gain') ? 'green' as const : 'yellow' as const },
          { label: 'Emergency', value: safety.resetRequired ? 'Reset required' : 'Clear', tone: safety.resetRequired ? 'red' as const : 'green' as const },
          { label: 'Traces', value: `${safety.traceCount} / ${safety.blockedTraceCount} blocked`, tone: safety.blockedTraceCount ? 'yellow' as const : 'green' as const },
          { label: 'Goal readiness', value: `${safety.goalCapabilityObjectiveReadinessScore}%`, tone: toneForReadinessScore(safety.goalCapabilityObjectiveReadinessScore) },
          { label: 'Readiness priority', value: readinessPriority, tone: safety.goalCapabilityObjectiveReadinessGoalAreaPriorities.length ? 'blue' as const : 'muted' as const },
          { label: 'Top hotspot', value: hotspotText, tone: safety.goalCapabilityObjectiveReadinessHotspots.length ? 'yellow' as const : 'green' as const },
          { label: 'Top blocked goal-area', value: blockedAreaSummary(safety.goalCapabilityObjectiveReadinessByGoalArea), tone: safety.goalCapabilityObjectiveReadinessByGoalArea && Object.keys(safety.goalCapabilityObjectiveReadinessByGoalArea).length ? 'yellow' as const : 'green' as const },
          { label: 'Proof blocked goal-area', value: proofBlockedAreaText, tone: safety.proofGoalCapabilityObjectiveReadinessByGoalArea && Object.keys(safety.proofGoalCapabilityObjectiveReadinessByGoalArea).length ? 'yellow' as const : 'green' as const },
          { label: 'Proof hotspot', value: proofHotspotText, tone: safety.proofGoalCapabilityObjectiveReadinessHotspots.length ? 'yellow' as const : 'green' as const },
          { label: 'Proof priority', value: proofReadinessPriority, tone: safety.proofGoalCapabilityObjectiveReadinessGoalAreaPriorities.length ? 'blue' as const : 'muted' as const },
        ].map(row => (
          <div className="rounded-[5px] border border-line bg-row px-1.5 py-1" key={row.label}>
            <div className="truncate text-[8px] uppercase tracking-[0.06em] text-dim">{row.label}</div>
            <div className={`truncate text-[9px] font-bold ${textClassForTone(row.tone)}`}>{row.value}</div>
          </div>
        ))}
      </div>
      <div className="mt-2 truncate text-[10px] text-muted">
        Blocked: {safety.blockedSurfaces.slice(0, 4).join(', ') || 'none'}
      </div>
    </Panel>
  );
}

function DecisionTracePanel({ traces }: { traces: DecisionTraceRecord[] }) {
  const rows = traces.slice(0, 4);
  return (
    <Panel title="Decision Traces">
      <div className="space-y-1">
        {rows.map(trace => (
          <div className="grid grid-cols-[38px_1fr_62px] items-center gap-2 border-b border-line pb-1 text-[9px]" key={trace.decisionId}>
            <span className="font-mono text-dim">{trace.createdAt}</span>
            <span className="truncate text-text">{trace.actionType} {trace.delta ? `· ${trace.delta} dB` : ''}</span>
            <StatusBadge tone={trace.tone}>{trace.blockedReason ? 'blocked' : trace.proofStatus}</StatusBadge>
          </div>
        ))}
        {!rows.length && <div className="rounded-[5px] border border-line bg-row px-2 py-2 text-[10px] text-muted">No backend traces yet.</div>}
      </div>
    </Panel>
  );
}

function toneForReplayStatus(status: string): Tone {
  const normalized = status.toLowerCase();
  if (normalized.includes('fail') || normalized.includes('blocked')) return 'red';
  if (normalized.includes('pass') || normalized.includes('clear') || normalized.includes('complete')) return 'green';
  if (normalized.includes('running') || normalized.includes('requested')) return 'blue';
  return 'yellow';
}

function ReplayEvidencePanel({
  traces,
  replayRuns,
  channelIntelligence,
  onStartReplay
}: {
  traces: DecisionTraceRecord[];
  replayRuns: ReplayRunRecord[];
  channelIntelligence: ChannelIntelligenceRecord[];
  onStartReplay: () => void;
}) {
  const latestRun = replayRuns[0];
  const latestTrace = traces[0];
  const channel = channelIntelligence.find(row => row.recentTraceCount > 0) ?? channelIntelligence[0];

  return (
    <Panel
      title="Replay Engine / Evidence"
      action={<button className="am-button py-1 text-[9px] text-audio" onClick={onStartReplay} type="button">Start Offline</button>}
    >
      <div className="flex items-center justify-between gap-2 rounded-[5px] border border-line bg-row px-1.5 py-1">
        <div className="min-w-0">
          <div className="truncate text-[8px] uppercase tracking-[0.06em] text-dim">Latest run</div>
          <div className="truncate text-[9px] font-bold text-text">{latestRun?.runId ?? 'No replay yet'}</div>
        </div>
        <StatusBadge tone={toneForReplayStatus(latestRun?.status ?? 'idle')}>{latestRun?.status ?? 'idle'}</StatusBadge>
      </div>
      <div className="mt-1 grid grid-cols-2 gap-1">
        <div className="rounded-[5px] border border-line bg-row px-1.5 py-1">
          <div className="truncate text-[8px] uppercase tracking-[0.06em] text-dim">Policy</div>
          <div className="truncate text-[9px] font-bold text-muted">
            {latestRun?.offlineOnly === false ? 'review' : 'offline only'} ·{' '}
            <span className={latestRun?.liveWrite ? 'text-danger' : 'text-success'}>
              {latestRun?.liveWrite ? 'live write' : 'no live write'}
            </span>
          </div>
        </div>
        <div className="rounded-[5px] border border-line bg-row px-1.5 py-1">
          <div className="truncate text-[8px] uppercase tracking-[0.06em] text-dim">Trace</div>
          <div className="truncate text-[9px] font-bold text-text">
            {latestRun ? `${latestRun.traceCount} rows` : latestTrace ? `${latestTrace.actionType} ${latestTrace.delta ? `${latestTrace.delta} dB` : ''}` : 'No decision trace'}
          </div>
        </div>
      </div>
      <div className="mt-1 flex items-center justify-between gap-2 text-[9px]">
        <span className="truncate text-text">{channel ? `${channel.channel} ${channel.name}` : 'No channel evidence'}</span>
        <StatusBadge tone={channel?.liveApplyAllowed ? 'yellow' : 'blue'}>{channel?.gainRideState ?? 'shadow'}</StatusBadge>
      </div>
      {latestRun && (
        <div className="mt-0.5 truncate text-[8px] text-muted">
          Report: {latestRun.reportPath || 'pending'}
        </div>
      )}
    </Panel>
  );
}

function SupervisedApplyModal({
  task,
  ui,
  evidence,
  applyPlan,
  integrationBundleBlockerNotice,
  integrationBundleApplyBlocked,
  onClose,
  onConfirm
}: {
  task: TaskRow;
  ui: UiState;
  evidence: SafetyEvidenceItem[];
  applyPlan: ApplyPlanStep[];
  integrationBundleBlockerNotice: string;
  integrationBundleApplyBlocked: boolean;
  onClose: () => void;
  onConfirm: () => void;
}) {
  const proposal = task.proposal;
  const canRequestBackendGate = ui.mode === 'Supervised'
    && ui.backend.canApplyToConsole
    && Boolean(task.canApply ?? true)
    && !task.reviewOnly
    && !integrationBundleApplyBlocked;
  const approvalLabel = ui.mode !== 'Supervised'
    ? 'Blocked by mode'
    : integrationBundleApplyBlocked
      ? 'Blocked by integration bundle'
    : canRequestBackendGate
      ? 'Backend supervised gate'
      : 'Local prototype approval only';
  const approvalTone: Tone = ui.mode !== 'Supervised' || integrationBundleApplyBlocked ? 'red' : canRequestBackendGate ? 'yellow' : 'blue';
  const applyRequest = canRequestBackendGate
    ? 'Ready to request'
    : ui.mode === 'Supervised' && !integrationBundleApplyBlocked
      ? 'UI-only simulation'
      : 'Blocked';
  const backendResult = ui.backend.proposalApplyResult;
  const resultMatches = Boolean(
    backendResult
    && (!backendResult.proposalId || backendResult.proposalId === task.proposalId || backendResult.proposalId === task.id)
  );
  const applyRequestLabel = resultMatches ? backendResult?.applyRequest ?? applyRequest : applyRequest;
  const readbackLabel = resultMatches ? backendResult?.readbackStatus ?? 'Required after backend apply' : 'Required after backend apply';
  const rollbackLabel = resultMatches ? backendResult?.rollbackStatus ?? 'Snapshot required' : 'Snapshot required';
  const backendResultTone = resultMatches ? backendResult?.tone ?? approvalTone : approvalTone;

  return (
    <div className="absolute inset-0 z-40 grid place-items-center bg-[#05080d]/80 p-5 backdrop-blur-sm" role="dialog" aria-label="Supervised Apply Proposal">
      <div className="w-full max-w-[760px] rounded-[8px] border border-warn/45 bg-[#0d151e] p-4 shadow-[0_18px_60px_rgba(0,0,0,.55)]">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <div className="am-label text-warn">Supervised Apply Proposal</div>
            <h2 className="mt-1 text-xl font-black text-text">{task.title}</h2>
            <p className="mt-1 text-[11px] text-muted">{task.id} · {task.agent} · no direct OSC/live write transport from frontend</p>
          </div>
          <StatusBadge tone={approvalTone}>{approvalLabel}</StatusBadge>
        </div>
        {integrationBundleBlockerNotice && (
          <div className="mb-3 rounded-[7px] border border-danger/35 bg-danger/10 px-3 py-2 text-[10px] leading-4 text-danger">
            {integrationBundleBlockerNotice}
          </div>
        )}
        <div className="grid grid-cols-[1.05fr_0.95fr] gap-3">
          <div className="space-y-3">
            <div className="rounded-[7px] border border-line bg-row p-3">
              <div className="am-label mb-2">Proposal Target</div>
              <div className="grid grid-cols-[92px_1fr] gap-y-2 text-[11px]">
                <span className="text-muted">Target</span><b className="truncate text-text">{task.target ?? task.title}</b>
                <span className="text-muted">Parameter</span><b className="truncate text-text">{proposal.parameter}</b>
                <span className="text-muted">Old value</span><b className="truncate text-muted">Current console snapshot</b>
                <span className="text-muted">New value</span><b className="truncate text-audio">{proposal.delta}</b>
                <span className="text-muted">Rollback</span><b className="truncate text-text">{proposal.rollback}</b>
              </div>
            </div>
            <div className="rounded-[7px] border border-line bg-row p-3">
              <div className="am-label mb-2">Safety Evidence</div>
              <div className="grid grid-cols-3 gap-2">
                {evidence.map(row => (
                  <div className="rounded-[5px] border border-line bg-panel2 px-2 py-1" key={`${row.label}-${row.value}`}>
                    <div className="truncate text-[8px] uppercase tracking-[0.08em] text-dim">{row.label}</div>
                    <div className={`truncate text-[10px] font-bold ${textClassForTone(row.tone)}`}>{row.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="space-y-3">
            <div className="rounded-[7px] border border-line bg-row p-3">
              <div className="am-label mb-2">Apply Gate</div>
              <div className="space-y-2">
                {[
                  { label: 'Required approval', value: approvalLabel, tone: approvalTone },
                  { label: 'Apply request', value: applyRequestLabel, tone: backendResultTone },
                  { label: 'Readback status', value: readbackLabel, tone: resultMatches && readbackLabel.includes('verified') ? 'green' as const : 'yellow' as const },
                  { label: 'Rollback status', value: rollbackLabel, tone: resultMatches && rollbackLabel.includes('available') ? 'green' as const : resultMatches && rollbackLabel.includes('failed') ? 'red' as const : 'yellow' as const },
                  { label: 'Final readiness', value: ui.backend.statusLabel, tone: ui.backend.statusTone },
                  { label: 'Direct writes', value: 'Disabled', tone: 'red' as const }
                ].map(row => (
                  <div className="flex items-center justify-between border-b border-line pb-2 text-[10px]" key={row.label}>
                    <span className="text-muted">{row.label}</span>
                    <StatusBadge tone={row.tone}>{row.value}</StatusBadge>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-[7px] border border-line bg-row p-3">
              <div className="am-label mb-2">Gate Sequence</div>
              <div className="grid grid-cols-5 gap-1">
                {applyPlan.map(step => (
                  <div className="rounded-[5px] border border-line bg-panel2 px-1.5 py-1" key={`${step.label}-${step.value}`}>
                    <div className="truncate text-[8px] uppercase tracking-[0.04em] text-dim">{step.label}</div>
                    <div className={`mt-1 truncate text-[9px] font-bold ${textClassForTone(step.tone)}`}>{step.value}</div>
                  </div>
                ))}
              </div>
            </div>
            {resultMatches && (
              <div className="rounded-[7px] border border-line bg-row p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <div className="am-label">Backend Result</div>
                  <StatusBadge tone={backendResultTone}>{backendResult?.status}</StatusBadge>
                </div>
                <p className="text-[10px] leading-4 text-muted">
                  {backendResult?.reason || 'Backend gate response received.'} · {backendResult?.updatedAt}
                </p>
              </div>
            )}
          </div>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <button className="am-button" onClick={onClose} type="button">Cancel</button>
          <button
            className="am-button border-warn/40 bg-warn/10 text-warn disabled:cursor-not-allowed disabled:opacity-60"
            disabled={!canRequestBackendGate}
            onClick={onConfirm}
            type="button"
          >
            {canRequestBackendGate ? 'Request Supervised Apply' : 'Apply Blocked'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function TaskQueueWorkspace({
  taskRows,
  selectedTask,
  ui,
  actions,
  policy,
  evidence,
  applyPlan,
  runtime,
  decisionTraces,
  safetyMonitor,
  replayRuns,
  channelIntelligence
}: {
  taskRows: TaskRow[];
  selectedTask: TaskRow;
  ui: UiState;
  actions: UiActions;
  policy: DecisionPolicy;
  evidence: SafetyEvidenceItem[];
  applyPlan: ApplyPlanStep[];
  runtime: DirectorRuntime;
  decisionTraces: DecisionTraceRecord[];
  safetyMonitor: SafetyMonitorState;
  replayRuns: ReplayRunRecord[];
  channelIntelligence: ChannelIntelligenceRecord[];
}) {
  const modalTask = ui.supervisedProposalTaskId
    ? taskRows.find(task => task.id === ui.supervisedProposalTaskId)
    : undefined;
  const modalEvidence = modalTask ? safetyEvidenceForTask(modalTask, ui.mode) : evidence;
  const modalApplyPlan = modalTask ? applyPlanForTask(modalTask, ui.mode) : applyPlan;
  const blockedReasonMessages = directorBlockedReasonMessages(runtime.blockedReasons);
  const integrationBundleApplyBlocked = blockedReasonMessages.length > 0;
  const taskQueueBlockerNotice = blockedReasonMessages.length > 0 && taskRows.length > 0
    ? `Promotion planning blocked. Task Queue evidence remains review-only: ${blockedReasonMessages[0]}`
    : '';
  const supervisedApplyBlockerNotice = blockedReasonMessages.length > 0 && modalTask
    ? `Promotion planning blocked. Supervised apply remains blocked: ${blockedReasonMessages[0]}`
    : '';
  const guardedConfirmSupervisedApply = (taskId: string) => {
    if (integrationBundleApplyBlocked) return;
    actions.confirmSupervisedApply(taskId);
  };

  return (
    <div className="relative grid h-full grid-rows-[auto_1fr_118px] gap-2">
      <div className="space-y-2">
        <div className="grid grid-cols-5 gap-2">
          {taskQueueMetrics(taskRows, integrationBundleApplyBlocked).map(metric => <MetricCard metric={metric} key={metric.label} />)}
        </div>
        {taskQueueBlockerNotice && (
          <div className="rounded-[6px] border border-danger/35 bg-danger/10 px-2 py-1.5 text-[10px] leading-4 text-danger">
            {taskQueueBlockerNotice}
          </div>
        )}
      </div>
      <DataTable
        rows={taskRows}
        selectedRow={selectedTask}
        onSelect={task => actions.selectTask(task.id)}
        rowKey={task => task.id}
        searchable
        searchPlaceholder="Search task, target, agent..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'High', predicate: task => task.priority === 'High' },
          { label: 'Review only', predicate: task => integrationBundleApplyBlocked || Boolean(task.reviewOnly || task.canApply === false) },
          { label: 'Pending', predicate: task => task.status === 'Pending' },
          { label: 'Applied', predicate: task => task.status === 'Applied' }
        ]}
        getSearchText={task => `${task.title} ${task.agent} ${task.target} ${task.status} ${task.priority} ${task.reason ?? ''}`}
        columns={[
          { key: 'priority', label: 'Priority', width: '0.75fr', render: task => <StatusBadge tone={toneForPriority(task.priority)}>{task.priority}</StatusBadge> },
          {
            key: 'title',
            label: 'Task',
            width: '1.55fr',
            render: task => (
              <span className="flex min-w-0 items-center gap-2">
                <span className="truncate">{task.title}</span>
                {(integrationBundleApplyBlocked || task.reviewOnly) && (
                  <StatusBadge tone="red">{integrationBundleApplyBlocked ? 'Blocked' : 'Review'}</StatusBadge>
                )}
              </span>
            )
          },
          { key: 'agent', label: 'Created By', width: '1.15fr' },
          { key: 'target', label: 'Target / Move', width: '1fr', render: task => <span className="truncate"><span className="text-text">{task.target}</span><span className="ml-2 font-mono text-audio">{task.proposal.delta}</span></span> },
          { key: 'status', label: 'Status', width: '0.9fr', render: task => <StatusBadge tone={toneFromState(task.status)}>{task.status}</StatusBadge> },
          { key: 'time', label: 'Last', width: '0.55fr' },
          {
            key: 'actions',
            label: 'Actions',
            width: '90px',
            render: task => (
              <ActionCluster
                actions={integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly ? ['Detail', 'Reject'] : ['Detail', 'Apply', 'Reject']}
                onAction={action => {
                  if (action === 'Apply') requestTaskApply(task, ui, actions, integrationBundleApplyBlocked);
                  else if (action === 'Reject') actions.setTaskStatus(task.id, 'Rejected');
                  else focusTaskDetail(actions, task);
                }}
              />
            )
          }
        ]}
      />
      <div className="grid grid-cols-[1fr_1fr_1fr_1.05fr_1.25fr] gap-2">
        <ApplyPlanPanel plan={applyPlan} />
        <SafetyEvidencePanel evidence={evidence} />
        <OperatorDecisionPanel ui={ui} policy={policy} />
        <SafetyMonitorPanel safety={safetyMonitor} />
        <ReplayEvidencePanel
          channelIntelligence={channelIntelligence}
          onStartReplay={actions.startReplayRun}
          replayRuns={replayRuns}
          traces={decisionTraces}
        />
      </div>
      {modalTask && (
        <SupervisedApplyModal
          applyPlan={modalApplyPlan}
          evidence={modalEvidence}
          integrationBundleApplyBlocked={integrationBundleApplyBlocked}
          integrationBundleBlockerNotice={supervisedApplyBlockerNotice}
          onClose={actions.closeSupervisedProposal}
          onConfirm={() => guardedConfirmSupervisedApply(modalTask.id)}
          task={modalTask}
          ui={ui}
        />
      )}
    </div>
  );
}
