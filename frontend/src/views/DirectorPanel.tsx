import { DirectorAvatar, Panel, StatusBadge } from '../components/prototype/AutomixerComponents';
import type { DirectorRuntime, Task, UiActions } from '../types';
import { TechnicalChecklist } from './viewHelpers';

export function RightDirectorPanel({
  actions,
  taskRows,
  runtime
}: {
  actions?: UiActions;
  taskRows: Task[];
  runtime: DirectorRuntime;
}) {
  return (
    <div className="flex h-full flex-col gap-2">
      <Panel>
        <div className="flex items-center gap-4">
          <DirectorAvatar />
          <div className="min-w-0">
            <h3 className="text-[15px] font-bold text-text">Automixer Director</h3>
            <p className="text-[10px] text-muted">Main management agent</p>
            <div className="mt-2 flex items-center gap-2">
              <StatusBadge tone="green">Online</StatusBadge>
              <span className="rounded border border-ai/30 bg-ai/10 px-1.5 py-0.5 text-[9px] font-bold uppercase text-ai">v0.8.7</span>
            </div>
          </div>
        </div>
      </Panel>
      <Panel title="Current Goal">
        <p className="text-[11px] leading-5 text-muted">{runtime.currentGoal}</p>
        <button className="am-button mt-3">Goal Details</button>
      </Panel>
      <Panel title="Runtime Guard">
        <div className="mb-3 grid grid-cols-2 gap-2 text-[10px]">
          <StatusBadge tone={runtime.readyTone}>{runtime.readyLabel}</StatusBadge>
          <StatusBadge tone={runtime.audioTone}>{runtime.audioLabel}</StatusBadge>
          <StatusBadge tone={runtime.proposalQueueTone}>{runtime.proposalQueueLabel}</StatusBadge>
          <StatusBadge tone={runtime.manualGateTone}>{runtime.manualGateLabel}</StatusBadge>
        </div>
        <TechnicalChecklist rows={[
          { label: 'Live write policy', value: runtime.liveWritePolicy, tone: runtime.manualGateTone },
          ...runtime.statusItems
        ]} />
        {runtime.blockedReasons.length > 0 && (
          <div className="mt-2 rounded-[5px] border border-warn/35 bg-warn/10 px-2 py-1.5 text-[10px] text-warn">
            {runtime.blockedReasons.slice(0, 2).join(' · ')}
          </div>
        )}
      </Panel>
      <Panel title="Hardware Proof / Evidence">
        <TechnicalChecklist rows={runtime.hardwareProofItems} />
        {runtime.resetRequired && (
          <div className="mt-2 rounded-[5px] border border-danger/40 bg-danger/10 px-2 py-1.5 text-[10px] text-danger">
            Emergency reset requires explicit operator action. Automation remains frozen until reviewed.
          </div>
        )}
      </Panel>
      <Panel title="Recent Decisions" className="flex-1">
        <div className="space-y-2">
          {taskRows.slice(0, 5).map(task => (
            <div className="grid grid-cols-[38px_1fr_52px] gap-2 border-b border-line pb-2 text-[10px]" key={task.id}>
              <span className="font-mono text-muted">{task.time}</span>
              <span className="truncate text-text">{task.title}</span>
              <span className="text-right text-online">{task.status}</span>
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="Agent Suggestions">
        <div className="space-y-3 text-[10px]">
          <div>
            <div className="text-text">Lead Vocal: add presence safely</div>
            <div className="text-muted">+1.5 dB @ 12 kHz</div>
            <div className="mt-2 flex gap-2"><button className="am-button text-online" onClick={() => actions?.pushToast('Suggestion accepted locally. No mixer write sent.', 'green')}>Accept</button><button className="am-button text-danger" onClick={() => actions?.pushToast('Suggestion rejected locally.', 'red')}>Reject</button></div>
          </div>
          <div>
            <div className="text-text">Kick: reduce muddiness</div>
            <div className="text-muted">-2 dB @ 250 Hz</div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
