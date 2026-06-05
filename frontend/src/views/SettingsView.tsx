import { DataTable, Knob, Panel, StatusBadge } from '../components/prototype/AutomixerComponents';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { ModuleStatusRecord, TelemetryAlertsSnapshot, Tone, UiActions, UiState } from '../types';
import { tableView, TechnicalChecklist } from './viewHelpers';

const settingsSections = [
  'Backend Policy',
  'Runtime Guard',
  'Module Status',
  'Mix Defaults',
  'Mode Policy',
  'Write Policy',
  'Audit',
  'Transport'
];

const fallbackModuleStatuses: ModuleStatusRecord[] = [
  {
    moduleId: 'product_operator_state',
    label: 'Product Operator State',
    domain: 'product_layer',
    status: 'mock',
    entrypoint: 'get_dashboard_snapshot / get_decision_queue',
    module: 'backend/operator_product_state.py',
    uiSurface: 'Control Center, Task Queue, Settings',
    canWriteLive: false,
    liveWriteKinds: [],
    pipelineProfiles: ['main_live_assist'],
    blockedReason: '',
    nextStep: 'Connect backend to load the canonical module status map.',
    tone: 'blue'
  },
  {
    moduleId: 'supervised_manual_fader_gain',
    label: 'Supervised Manual Fader/Gain',
    domain: 'gain_fader',
    status: 'mock',
    entrypoint: 'apply_operator_proposal -> supervised manual gate',
    module: 'backend/server.py',
    uiSurface: 'Task Queue supervised apply',
    canWriteLive: true,
    liveWriteKinds: ['fader', 'gain'],
    pipelineProfiles: ['main_live_assist'],
    blockedReason: '',
    nextStep: 'Connect backend for live policy and proof state.',
    tone: 'yellow'
  },
  {
    moduleId: 'legacy_reset_surfaces',
    label: 'Legacy Reset Surfaces',
    domain: 'quarantine',
    status: 'mock_blocked',
    entrypoint: 'blocked from cockpit',
    module: 'backend/server.py',
    uiSurface: 'Hidden from live cockpit',
    canWriteLive: false,
    liveWriteKinds: [],
    pipelineProfiles: ['quarantine'],
    blockedReason: 'direct live reset is not an MVP apply path',
    nextStep: 'Use proposal/supervised proof flow instead of direct reset buttons.',
    tone: 'red'
  }
];

function connectionToneFor(state: UiState['backend']['connectionState']): Tone {
  if (state === 'connected') return 'green';
  if (state === 'connecting') return 'yellow';
  if (state === 'error') return 'red';
  return 'blue';
}

function moduleStatusRows(data: OperatorViewData) {
  return data.moduleStatuses.length ? data.moduleStatuses : fallbackModuleStatuses;
}

function writeScopeLabel(row: ModuleStatusRecord) {
  if (!row.canWriteLive) return 'blocked';
  return row.liveWriteKinds.length ? row.liveWriteKinds.join(', ') : 'live';
}

function telemetryAlertTone(alerts: TelemetryAlertsSnapshot): Tone {
  if (alerts.liveWrite || alerts.generatesArtifacts) return 'red';
  if (alerts.criticalCount > 0) return 'red';
  if (alerts.warningCount > 0) return 'yellow';
  if (alerts.alertCount > 0) return 'blue';
  return alerts.available ? 'green' : 'muted';
}

function telemetryAlertRows(alerts: TelemetryAlertsSnapshot) {
  const safetyOk = alerts.readOnly && !alerts.liveWrite && !alerts.generatesArtifacts;
  return [
    { label: 'Artifacts', value: alerts.available ? 'Available' : 'Missing', tone: alerts.available ? 'blue' as const : 'muted' as const },
    { label: 'Alerts', value: String(alerts.alertCount), tone: telemetryAlertTone(alerts) },
    { label: 'Critical', value: String(alerts.criticalCount), tone: alerts.criticalCount ? 'red' as const : 'green' as const },
    { label: 'Warning', value: String(alerts.warningCount), tone: alerts.warningCount ? 'yellow' as const : 'green' as const },
    { label: 'Snapshot', value: safetyOk ? 'Read-only' : 'Review', tone: safetyOk ? 'green' as const : 'red' as const },
    { label: 'Latest session', value: alerts.latestSessionId || 'none', tone: alerts.latestSessionId ? 'blue' as const : 'muted' as const }
  ];
}

function readableStatus(value: string) {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function modeGateRow(rows: ModuleStatusRecord[], mode: string) {
  return rows.find(row => row.moduleId === `mode_gate:${mode}`);
}

function modeGateWriteKinds(row?: ModuleStatusRecord) {
  if (!row) return [];
  return row.pipelineProfiles
    .filter(profile => profile.startsWith('current_write:'))
    .map(profile => profile.replace('current_write:', ''))
    .filter(Boolean);
}

function modePolicyRows(ui: UiState, modules: ModuleStatusRecord[], writeScope: string) {
  const supervised = modeGateRow(modules, 'supervised');
  const auto = modeGateRow(modules, 'auto');
  const supervisedWriteKinds = modeGateWriteKinds(supervised);
  return [
    { label: 'Current mode', value: ui.mode, tone: ui.mode === 'Supervised' ? 'yellow' as const : ui.mode === 'Auto' ? 'red' as const : 'blue' as const },
    { label: 'Supervised gate', value: supervised ? readableStatus(supervised.status) : 'Unknown', tone: supervised?.tone ?? 'yellow' as const },
    { label: 'Supervised scope', value: supervisedWriteKinds.length ? supervisedWriteKinds.join(', ') : writeScope, tone: supervisedWriteKinds.length ? 'yellow' as const : 'red' as const },
    { label: 'Full Auto', value: auto ? readableStatus(auto.status) : 'Unknown', tone: auto?.tone ?? 'red' as const }
  ];
}

function profileValue(row: ModuleStatusRecord | undefined, prefix: string) {
  if (!row) return '';
  return row.pipelineProfiles
    .find(profile => profile.startsWith(prefix))
    ?.slice(prefix.length) ?? '';
}

function compactReadSurfaceGapLabel(gapIds: string[], gapCount: number | null) {
  if (!gapIds.length) {
    if (gapCount && gapCount > 0) return `${gapCount} surfaces unlisted`;
    return 'None listed';
  }
  const remaining = Math.max(0, (gapCount ?? gapIds.length) - 1);
  return remaining > 0 ? `${gapIds[0]} +${remaining}` : gapIds[0];
}

function proofEvidenceInspectorRows(modules: ModuleStatusRecord[]) {
  const proof = modules.find(row => row.moduleId === 'supervised_proof_evidence');
  const overlayRows = modules.filter(row => row.domain === 'autonomy_evidence_overlay');
  if (!proof) {
    return [
      { label: 'Proof evidence', value: 'Missing', tone: 'yellow' as const },
      { label: 'Proof freshness', value: 'No artifact', tone: 'yellow' as const },
      { label: 'Autonomy overlay', value: overlayRows.length ? `${overlayRows.length} rows` : 'Missing', tone: overlayRows.length ? 'yellow' as const : 'muted' as const },
      { label: 'Read-surface proof', value: 'Missing', tone: 'muted' as const },
      { label: 'Gap surfaces', value: 'None listed', tone: 'muted' as const },
      { label: 'Next read proof', value: 'No target', tone: 'muted' as const },
      { label: 'Read target gate', value: 'Missing', tone: 'muted' as const },
      { label: 'Raw mixer contract', value: 'Missing', tone: 'muted' as const },
      { label: 'Contract gap', value: 'Unknown', tone: 'muted' as const },
      { label: 'Contract authority', value: 'Unknown', tone: 'muted' as const },
      { label: 'Contract target', value: 'Missing', tone: 'muted' as const }
    ];
  }
  const stale = profileValue(proof, 'stale:') === 'true';
  const overlayStatus = profileValue(proof, 'overlay:') || 'unknown';
  const readSurfaceQualityAvailable = profileValue(proof, 'read_surface_quality:') === 'true';
  const readSurfaceGapText = profileValue(proof, 'read_surface_gaps:');
  const readSurfaceGapCount = Number.parseInt(readSurfaceGapText, 10);
  const hasReadSurfaceGapCount = Number.isFinite(readSurfaceGapCount);
  const fixtureTestText = profileValue(proof, 'fixture_tests:');
  const fixtureTestCount = Number.parseInt(fixtureTestText, 10);
  const hasFixtureTestCount = Number.isFinite(fixtureTestCount);
  const contractGapEnforced = profileValue(proof, 'contract_gap:') === 'true';
  const gapSurfaceIds = profileValue(proof, 'read_surface_gap_ids:')
    .split('|')
    .map(value => value.trim())
    .filter(Boolean);
  const nextReadbackTarget = profileValue(proof, 'next_readback_target:');
  const nextReadbackGateValue = profileValue(proof, 'next_readback_gate:');
  const nextReadbackTargetConsistent = nextReadbackGateValue === 'true';
  const rawContractAvailable = profileValue(proof, 'raw_contract_available:') === 'true';
  const rawContractStatus = profileValue(proof, 'raw_contract_status:') || 'missing';
  const rawContractGapRemains = profileValue(proof, 'raw_contract_gap:') === 'true';
  const rawContractLiveAuthority = profileValue(proof, 'raw_contract_live_authority:') === 'true';
  const rawContractTargetConsistent = profileValue(proof, 'raw_contract_target_gate:') === 'true';
  const readSurfaceGapValue = hasReadSurfaceGapCount
    ? `${readSurfaceGapCount} physical gaps`
    : 'Unknown';
  const readSurfaceGapTone = !readSurfaceQualityAvailable
    ? 'muted' as const
    : hasReadSurfaceGapCount && readSurfaceGapCount === 0
      ? 'green' as const
      : 'yellow' as const;
  const readSurfaceProofTone = !readSurfaceQualityAvailable
    ? 'muted' as const
    : hasReadSurfaceGapCount && readSurfaceGapCount === 0 && contractGapEnforced
      ? 'green' as const
      : 'yellow' as const;
  return [
    { label: 'Proof evidence', value: readableStatus(proof.status), tone: proof.tone },
    { label: 'Proof freshness', value: stale ? 'Stale' : 'Fresh', tone: stale ? 'yellow' as const : 'green' as const },
    { label: 'Autonomy overlay', value: `${readableStatus(overlayStatus)} (${overlayRows.length})`, tone: overlayRows.length ? 'blue' as const : 'yellow' as const },
    { label: 'Read-surface proof', value: readSurfaceQualityAvailable ? 'Metadata available' : 'Missing', tone: readSurfaceProofTone },
    { label: 'Read-surface gaps', value: readSurfaceGapValue, tone: readSurfaceGapTone },
    { label: 'Gap surfaces', value: compactReadSurfaceGapLabel(gapSurfaceIds, hasReadSurfaceGapCount ? readSurfaceGapCount : null), tone: gapSurfaceIds.length ? 'yellow' as const : 'muted' as const },
    { label: 'Next read proof', value: nextReadbackTarget || 'No target', tone: nextReadbackTarget ? 'yellow' as const : 'muted' as const },
    { label: 'Read target gate', value: nextReadbackTargetConsistent ? 'Consistent' : 'Mismatch', tone: readSurfaceQualityAvailable ? nextReadbackTargetConsistent ? 'green' as const : 'yellow' as const : 'muted' as const },
    { label: 'Fixture tests', value: hasFixtureTestCount ? `${fixtureTestCount} declared tests` : 'Unknown', tone: readSurfaceQualityAvailable ? 'blue' as const : 'muted' as const },
    { label: 'Contract/gap gate', value: contractGapEnforced ? 'Enforced' : 'Missing', tone: contractGapEnforced ? 'green' as const : 'yellow' as const },
    { label: 'Raw mixer contract', value: rawContractAvailable ? readableStatus(rawContractStatus) : 'Missing', tone: rawContractAvailable ? 'blue' as const : 'muted' as const },
    { label: 'Contract gap', value: rawContractGapRemains ? 'Physical proof missing' : 'No gap reported', tone: rawContractGapRemains ? 'yellow' as const : 'green' as const },
    { label: 'Contract authority', value: rawContractLiveAuthority ? 'Live authority claimed' : 'No live authority', tone: rawContractLiveAuthority ? 'red' as const : 'green' as const },
    { label: 'Contract target', value: rawContractTargetConsistent ? 'Consistent' : 'Mismatch', tone: rawContractAvailable ? rawContractTargetConsistent ? 'green' as const : 'yellow' as const : 'muted' as const }
  ];
}

function ModuleStatusTable({
  rows,
  onFocus
}: {
  rows: ModuleStatusRecord[];
  onFocus: () => void;
}) {
  return (
    <div className="grid min-h-0 grid-rows-[40px_1fr] gap-2">
      <div className="am-panel flex items-center justify-between px-3">
        <div>
          <div className="am-label">Module Status</div>
          <div className="text-[9px] text-muted">entrypoint / UI surface / live write scope</div>
        </div>
        <button className="am-button" onClick={onFocus} type="button">Focus</button>
      </div>
      <DataTable
        rows={rows}
        rowKey={row => row.moduleId}
        searchable
        searchPlaceholder="Search module, entrypoint, UI..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Live', predicate: row => row.canWriteLive },
          { label: 'Write Surface', predicate: row => row.domain === 'write_surface' },
          { label: 'Proof', predicate: row => row.domain === 'supervised_proof_evidence' || row.domain === 'autonomy_evidence_overlay' || row.domain === 'read_surface_contract' },
          { label: 'Review', predicate: row => row.status.includes('proposal') || !row.canWriteLive },
          { label: 'Blocked', predicate: row => row.status.includes('blocked') || row.status.includes('quarantined') || row.pipelineProfiles.includes('quarantine') }
        ]}
        getSearchText={row => `${row.label} ${row.moduleId} ${row.status} ${row.domain} ${row.entrypoint} ${row.uiSurface} ${row.nextStep}`}
        columns={[
          { key: 'label', label: 'Module', width: '1.05fr', render: row => <span title={row.moduleId}>{row.label}</span> },
          { key: 'status', label: 'Status', width: '0.72fr', render: row => <StatusBadge tone={row.tone}>{row.status}</StatusBadge> },
          { key: 'uiSurface', label: 'UI Surface', width: '1.1fr' },
          { key: 'write', label: 'Write', width: '0.72fr', render: row => <StatusBadge tone={row.canWriteLive ? 'yellow' : 'red'}>{writeScopeLabel(row)}</StatusBadge> },
          { key: 'nextStep', label: 'Next Step', width: '1.45fr' }
        ]}
      />
    </div>
  );
}

function SettingsInspector({
  section,
  ui,
  data,
  writeScope
}: {
  section: string;
  ui: UiState;
  data: OperatorViewData;
  writeScope: string;
}) {
  const backend = ui.backend;
  const connectionTone = connectionToneFor(backend.connectionState);
  const modules = moduleStatusRows(data);
  const telemetryAlerts = data.telemetryAlerts;
  const liveModuleCount = modules.filter(row => row.canWriteLive).length;
  const reviewOnlyCount = modules.filter(row => !row.canWriteLive).length;
  const quarantinedCount = modules.filter(row => row.status.includes('quarantined') || row.pipelineProfiles.includes('quarantine')).length;
  const modeRows = modePolicyRows(ui, modules, writeScope);
  const rowsBySection: Record<string, { label: string; value: string; tone?: Tone }[]> = {
    'Backend Policy': [
      { label: 'Connection', value: backend.statusLabel, tone: backend.statusTone },
      { label: 'Data state', value: backend.dataStateLabel, tone: backend.dataStateTone },
      { label: 'Source', value: backend.sourceLabel, tone: backend.statusTone },
      { label: 'Last update', value: backend.lastUpdated === 'mock' ? 'mock' : backend.lastUpdated.slice(11, 19), tone: 'blue' }
    ],
    'Runtime Guard': [
      { label: 'Ready', value: data.directorRuntime.readyLabel, tone: data.directorRuntime.readyTone },
      { label: 'Audio', value: data.directorRuntime.audioLabel, tone: data.directorRuntime.audioTone },
      { label: 'Queue', value: data.directorRuntime.proposalQueueLabel, tone: data.directorRuntime.proposalQueueTone },
      { label: 'Manual gate', value: data.directorRuntime.manualGateLabel, tone: data.directorRuntime.manualGateTone }
    ],
    'Module Status': [
      { label: 'Modules', value: String(modules.length), tone: modules.length ? 'blue' : 'yellow' },
      { label: 'Live executors', value: String(liveModuleCount), tone: liveModuleCount === 1 ? 'green' : 'yellow' },
      { label: 'Review/status only', value: String(reviewOnlyCount), tone: reviewOnlyCount ? 'yellow' : 'green' },
      { label: 'Quarantine', value: String(quarantinedCount), tone: quarantinedCount ? 'red' : 'green' },
      { label: 'Allowed writes', value: writeScope, tone: backend.writeScope.length ? 'yellow' : 'red' },
      ...proofEvidenceInspectorRows(modules)
    ],
    'Mix Defaults': [
      { label: 'LUFS target', value: '-14 LUFS', tone: 'blue' },
      { label: 'True peak', value: '-1 dBTP', tone: 'green' },
      { label: 'Fader ceiling', value: '0 dB', tone: 'yellow' },
      { label: 'Snapshot', value: 'Required', tone: 'yellow' }
    ],
    'Mode Policy': modeRows,
    'Write Policy': [
      { label: 'Apply path', value: backend.applyPath, tone: backend.canApplyToConsole ? 'yellow' : 'blue' },
      { label: 'Console apply', value: backend.canApplyToConsole ? 'Supervised' : 'Blocked', tone: backend.canApplyToConsole ? 'yellow' : 'red' },
      { label: 'Scope', value: writeScope, tone: backend.writeScope.length ? 'yellow' : 'red' },
      { label: 'Direct OSC', value: 'Unavailable', tone: 'red' }
    ],
    Audit: [
      { label: 'Decision log', value: data.logs.length ? 'On' : 'Empty', tone: data.logs.length ? 'green' : 'yellow' },
      { label: 'Console diff', value: 'Read-only', tone: 'blue' },
      { label: 'Task rows', value: String(data.tasks.length), tone: 'purple' },
      { label: 'Telemetry alerts', value: String(telemetryAlerts.alertCount), tone: telemetryAlertTone(telemetryAlerts) },
      { label: 'Alert source', value: data.readModelSources.telemetryAlerts, tone: telemetryAlerts.available ? 'blue' : 'muted' },
      { label: 'Stale snapshot', value: backend.staleSnapshot ? 'Yes' : 'No', tone: backend.staleSnapshot ? 'yellow' : 'green' }
    ],
    Transport: [
      { label: 'Websocket', value: backend.connectionState, tone: connectionTone },
      { label: 'Product Layer', value: backend.dataStateLabel, tone: backend.dataStateTone },
      { label: 'Target', value: backend.connectionLabel, tone: connectionTone },
      { label: 'Live writes', value: 'No direct path', tone: 'red' }
    ]
  };
  const rows = rowsBySection[section] ?? rowsBySection['Backend Policy'];

  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Settings Inspector">
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="truncate text-[15px] font-bold text-text">{section}</h3>
          <StatusBadge tone={section === 'Write Policy' ? 'yellow' : section === 'Transport' ? connectionTone : 'blue'}>Read-only</StatusBadge>
        </div>
        <TechnicalChecklist rows={rows} />
      </Panel>
      <Panel title="Backend Message" className="flex-1">
        <p className="text-[11px] leading-5 text-muted">{backend.message}</p>
        <div className="mt-4 h-24"><Knob label="Confidence Gate" value={data.dashboardRuntime.aiConfidencePercent} /></div>
      </Panel>
      <Panel title="Telemetry Alerts">
        <TechnicalChecklist rows={telemetryAlertRows(telemetryAlerts)} />
        <div className="mt-3 space-y-2">
          {telemetryAlerts.alerts.slice(0, 2).map(alert => (
            <div className="rounded-[6px] border border-line bg-row p-2" key={alert.id}>
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="truncate text-[10px] font-bold text-text">{alert.rule}</span>
                <StatusBadge tone={alert.tone}>{alert.severity}</StatusBadge>
              </div>
              <div className="truncate text-[10px] text-muted">{alert.target} · {alert.time}</div>
              <div className="mt-1 max-h-8 overflow-hidden text-[10px] leading-4 text-muted">{alert.message}</div>
            </div>
          ))}
          {!telemetryAlerts.alerts.length && (
            <div className="rounded-[6px] border border-line bg-row p-2 text-[10px] text-muted">
              No generated alert artifacts are available for display.
            </div>
          )}
        </div>
      </Panel>
      <Panel title="Safety Rule">
        <TechnicalChecklist rows={[
          { label: 'OSC writes', value: 'Disabled', tone: 'red' },
          { label: 'Routing writes', value: 'Disabled', tone: 'red' },
          { label: 'Apply UX', value: 'Supervised only', tone: 'yellow' },
          { label: 'Module map', value: `${modules.length} rows`, tone: modules.length ? 'blue' : 'yellow' }
        ]} />
      </Panel>
    </div>
  );
}

export function Settings(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const backend = ui.backend;
  const runtime = data.directorRuntime;
  const selectedSection = settingsSections.includes(ui.selectedSettingsSection) ? ui.selectedSettingsSection : settingsSections[0];
  const writeScope = backend.writeScope.length ? backend.writeScope.join(', ') : 'none';
  const connectionTone = connectionToneFor(backend.connectionState);
  const modules = moduleStatusRows(data);
  const telemetryAlerts = data.telemetryAlerts;
  const modeRows = modePolicyRows(ui, modules, writeScope);
  const approvalLabel = backend.canApplyToConsole
    ? (backend.requiresApproval ? 'Required' : 'Not required')
    : 'No apply';
  const approvalTone: Tone = backend.canApplyToConsole
    ? (backend.requiresApproval ? 'yellow' : 'red')
    : 'red';

  return tableView('7. Settings', 'System, automation, and mix defaults', (
    <div className="grid h-full grid-rows-[38px_1fr_116px] gap-2">
      <div className="am-panel flex items-center justify-between px-3">
        <div className="flex gap-1 text-[10px] font-semibold uppercase text-muted">
          {settingsSections.slice(0, 6).map(section => (
            <button
              className={`rounded-[4px] border px-2 py-1 transition ${selectedSection === section ? 'border-audio bg-audio/10 text-audio' : 'border-transparent text-muted hover:text-text'}`}
              key={section}
              onClick={() => actions.selectSettingsSection(section)}
              type="button"
            >
              {section.replace('Backend ', '')}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <button className="am-button" onClick={actions.refreshBackend} type="button">Refresh Backend</button>
          <button className="am-button bg-audio/15 text-audio" onClick={actions.runBackendAnalysis} type="button">Run Analysis</button>
        </div>
      </div>
      <div className="grid grid-cols-[0.9fr_1.1fr_1fr] gap-2">
        <Panel title="Backend Policy" action={<button className="am-button" onClick={() => actions.selectSettingsSection('Backend Policy')} type="button">Focus</button>}>
          <div className="space-y-2">
            <div className="rounded-[6px] border border-line bg-row p-2">
              <div className="mb-2 flex items-center justify-between"><span className="text-[11px] font-bold text-text">Connection</span><StatusBadge tone={connectionTone}>{backend.statusLabel}</StatusBadge></div>
              <TechnicalChecklist rows={[
                { label: 'Source', value: backend.sourceLabel, tone: backend.statusTone },
                { label: 'Target', value: backend.connectionLabel, tone: connectionTone },
                { label: 'Data state', value: backend.dataStateLabel, tone: backend.dataStateTone },
                { label: 'Stale snapshot', value: backend.staleSnapshot ? 'Yes' : 'No', tone: backend.staleSnapshot ? 'yellow' : 'green' },
                { label: 'Last update', value: backend.lastUpdated === 'mock' ? 'mock' : backend.lastUpdated.slice(11, 19), tone: 'blue' }
              ]} />
            </div>
            <div className="rounded-[6px] border border-line bg-row p-2">
              <div className="mb-2 flex items-center justify-between"><span className="text-[11px] font-bold text-text">Operator Policy</span><StatusBadge tone={backend.canApplyToConsole ? 'yellow' : 'green'}>{backend.canApplyToConsole ? 'Gate' : 'Read'}</StatusBadge></div>
              <TechnicalChecklist rows={[
                { label: 'Live policy', value: backend.liveWritePolicy, tone: backend.canApplyToConsole ? 'yellow' : 'green' },
                { label: 'Apply path', value: backend.applyPath, tone: backend.canApplyToConsole ? 'yellow' : 'blue' },
                { label: 'Write scope', value: writeScope, tone: backend.writeScope.length ? 'yellow' : 'red' },
                { label: 'Approval', value: approvalLabel, tone: approvalTone }
              ]} />
            </div>
          </div>
        </Panel>
        <Panel title="Runtime Guard" action={<button className="am-button" onClick={() => actions.selectSettingsSection('Runtime Guard')} type="button">Focus</button>}>
          <div className="grid grid-cols-2 gap-x-6 gap-y-5">
            <Knob label="Safe Amount" value={62} />
            <Knob label="Confidence Gate" value={data.dashboardRuntime.aiConfidencePercent} />
            <Knob label="Auto Learn" value={0} />
            <Knob label="Rollback Sensitivity" value={78} />
          </div>
          <div className="mt-6 grid grid-cols-2 gap-2">
            <TechnicalChecklist rows={[
              { label: 'Ready state', value: runtime.readyLabel, tone: runtime.readyTone },
              { label: 'Audio capture', value: runtime.audioLabel, tone: runtime.audioTone },
              { label: 'Proposal queue', value: runtime.proposalQueueLabel, tone: runtime.proposalQueueTone }
            ]} />
            <TechnicalChecklist rows={[
              { label: 'Manual gate', value: runtime.manualGateLabel, tone: runtime.manualGateTone },
              { label: 'Auto writes', value: 'Disabled', tone: 'red' },
              { label: 'Direct OSC', value: 'Unavailable', tone: 'red' }
            ]} />
          </div>
        </Panel>
        <ModuleStatusTable rows={modules} onFocus={() => actions.selectSettingsSection('Module Status')} />
      </div>
      <div className="grid grid-cols-6 gap-2">
        <Panel title="Mix Defaults"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Mix Defaults')} type="button"><TechnicalChecklist rows={[{ label: 'LUFS target', value: '-14 LUFS', tone: 'blue' }, { label: 'Safety margin', value: 'High', tone: 'yellow' }]} /></button></Panel>
        <Panel title="Mode Policy"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Mode Policy')} type="button"><TechnicalChecklist rows={modeRows.slice(0, 2)} /></button></Panel>
        <Panel title="Write Policy"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Write Policy')} type="button"><TechnicalChecklist rows={[{ label: 'Console apply', value: backend.canApplyToConsole ? 'Supervised' : 'Blocked', tone: backend.canApplyToConsole ? 'yellow' : 'red' }, { label: 'Scope', value: writeScope, tone: backend.writeScope.length ? 'yellow' : 'red' }]} /></button></Panel>
        <Panel title="Module Status"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Module Status')} type="button"><TechnicalChecklist rows={[{ label: 'Modules', value: String(modules.length), tone: 'blue' }, { label: 'Live writes', value: String(modules.filter(row => row.canWriteLive).length), tone: 'yellow' }]} /></button></Panel>
        <Panel title="Audit"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Audit')} type="button"><TechnicalChecklist rows={[{ label: 'Decision log', value: data.logs.length ? 'On' : 'Empty', tone: data.logs.length ? 'green' : 'yellow' }, { label: 'Telemetry alerts', value: String(telemetryAlerts.alertCount), tone: telemetryAlertTone(telemetryAlerts) }, { label: 'Console diff', value: 'Read-only', tone: 'blue' }]} /></button></Panel>
        <Panel title="Transport"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Transport')} type="button"><TechnicalChecklist rows={[{ label: 'Websocket', value: backend.connectionState, tone: connectionTone }, { label: 'Product Layer', value: backend.dataStateLabel, tone: backend.dataStateTone }, { label: 'OSC writes', value: 'No direct path', tone: 'red' }]} /></button></Panel>
      </div>
    </div>
  ), <SettingsInspector data={data} section={selectedSection} ui={ui} writeScope={writeScope} />);
}
