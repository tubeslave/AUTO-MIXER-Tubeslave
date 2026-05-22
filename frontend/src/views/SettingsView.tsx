import { Knob, Panel, StatusBadge } from '../components/prototype/AutomixerComponents';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { Tone, UiActions, UiState } from '../types';
import { MiniBarRow, SettingRows, tableView, TechnicalChecklist } from './viewHelpers';

const settingsSections = [
  'Backend Policy',
  'Runtime Guard',
  'Mix Defaults',
  'Mode Policy',
  'Write Policy',
  'Audit',
  'Transport'
];

function connectionToneFor(state: UiState['backend']['connectionState']): Tone {
  if (state === 'connected') return 'green';
  if (state === 'connecting') return 'yellow';
  if (state === 'error') return 'red';
  return 'blue';
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
    'Mix Defaults': [
      { label: 'LUFS target', value: '-14 LUFS', tone: 'blue' },
      { label: 'True peak', value: '-1 dBTP', tone: 'green' },
      { label: 'Fader ceiling', value: '0 dB', tone: 'yellow' },
      { label: 'Snapshot', value: 'Required', tone: 'yellow' }
    ],
    'Mode Policy': [
      { label: 'Current mode', value: ui.mode, tone: ui.mode === 'Supervised' ? 'yellow' : ui.mode === 'Auto' ? 'red' : 'blue' },
      { label: 'Create proposals', value: backend.canCreateProposals ? 'Allowed' : 'Blocked', tone: backend.canCreateProposals ? 'green' : 'yellow' },
      { label: 'Auto apply', value: 'Disabled', tone: 'red' },
      { label: 'Analyze/Assist apply', value: 'Blocked', tone: 'red' }
    ],
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
      <Panel title="Safety Rule">
        <TechnicalChecklist rows={[
          { label: 'OSC writes', value: 'Disabled', tone: 'red' },
          { label: 'Routing writes', value: 'Disabled', tone: 'red' },
          { label: 'Apply UX', value: 'Supervised only', tone: 'yellow' }
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
  const approvalLabel = backend.canApplyToConsole
    ? (backend.requiresApproval ? 'Required' : 'Not required')
    : 'No apply';
  const approvalTone: Tone = backend.canApplyToConsole
    ? (backend.requiresApproval ? 'yellow' : 'red')
    : 'red';

  return tableView('7. Settings', 'System, automation, and mix defaults', (
    <div className="grid h-full grid-rows-[38px_1fr_108px] gap-2">
      <div className="am-panel flex items-center justify-between px-3">
        <div className="flex gap-1 text-[10px] font-semibold uppercase text-muted">
          {settingsSections.slice(0, 5).map(section => (
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
        <Panel title="Mix Defaults" action={<button className="am-button" onClick={() => actions.selectSettingsSection('Mix Defaults')} type="button">Focus</button>}>
          <SettingRows rows={['Default LUFS Target -14 LUFS', 'True Peak Limit -1 dBTP', 'Max Fader Ceiling 0 dB', 'Minimum Headroom 3 dB', 'Snapshot Before Apply On']} />
          <div className="mt-5 space-y-3">
            <MiniBarRow label="Vocal Priority" value="72%" meter={-14} tone="purple" />
            <MiniBarRow label="Drum Stability" value="84%" meter={-10} tone="green" />
            <MiniBarRow label="Safety Margin" value="High" meter={-18} tone="yellow" />
          </div>
        </Panel>
      </div>
      <div className="grid grid-cols-4 gap-2">
        <Panel title="Mode Policy"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Mode Policy')} type="button"><TechnicalChecklist rows={[{ label: 'Current mode', value: ui.mode, tone: backend.canApplyToConsole ? 'yellow' : 'blue' }, { label: 'Create proposals', value: backend.canCreateProposals ? 'Allowed' : 'Blocked', tone: backend.canCreateProposals ? 'green' : 'yellow' }]} /></button></Panel>
        <Panel title="Write Policy"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Write Policy')} type="button"><TechnicalChecklist rows={[{ label: 'Console apply', value: backend.canApplyToConsole ? 'Supervised' : 'Blocked', tone: backend.canApplyToConsole ? 'yellow' : 'red' }, { label: 'Scope', value: writeScope, tone: backend.writeScope.length ? 'yellow' : 'red' }]} /></button></Panel>
        <Panel title="Audit"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Audit')} type="button"><TechnicalChecklist rows={[{ label: 'Decision log', value: data.logs.length ? 'On' : 'Empty', tone: data.logs.length ? 'green' : 'yellow' }, { label: 'Console diff', value: 'Read-only', tone: 'blue' }]} /></button></Panel>
        <Panel title="Transport"><button className="mb-2 w-full text-left" onClick={() => actions.selectSettingsSection('Transport')} type="button"><TechnicalChecklist rows={[{ label: 'Websocket', value: backend.connectionState, tone: connectionTone }, { label: 'Product Layer', value: backend.dataStateLabel, tone: backend.dataStateTone }, { label: 'OSC writes', value: 'No direct path', tone: 'red' }]} /></button></Panel>
      </div>
    </div>
  ), <SettingsInspector data={data} section={selectedSection} ui={ui} writeScope={writeScope} />);
}
