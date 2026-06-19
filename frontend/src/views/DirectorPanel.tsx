import { DirectorAvatar, Panel, StatusBadge } from '../components/prototype/AutomixerComponents';
import {
  directorBlockedReasonMessages,
  directorDecisionRows,
  directorProposalRows,
  type DirectorProposalRow
} from '../components/dashboard/directorPanelModel';
import type { DirectorRuntime, Task, Tone, UiActions } from '../types';
import { TechnicalChecklist } from './viewHelpers';

function toneText(tone: Tone) {
  if (tone === 'red') return 'text-danger';
  if (tone === 'yellow') return 'text-warn';
  if (tone === 'purple') return 'text-ai';
  if (tone === 'blue') return 'text-audio';
  return 'text-online';
}

const fallbackSuggestionRows: DirectorProposalRow[] = [
  {
    id: 'mock-presence-suggestion',
    agent: 'Mock',
    title: 'Lead Vocal: add presence safely',
    detail: '+1.5 dB @ 12 kHz - UI fallback',
    status: 'Review',
    tone: 'purple',
    reviewMessage: 'Suggestion reviewed locally. No mixer write sent.',
    rejectMessage: 'Suggestion rejected locally. No mixer write sent.'
  },
  {
    id: 'mock-kick-suggestion',
    agent: 'Mock',
    title: 'Kick: reduce muddiness',
    detail: '-2 dB @ 250 Hz - UI fallback',
    status: 'Review',
    tone: 'blue',
    reviewMessage: 'Suggestion reviewed locally. No mixer write sent.',
    rejectMessage: 'Suggestion rejected locally. No mixer write sent.'
  }
];

export function RightDirectorPanel({
  actions,
  taskRows,
  runtime
}: {
  actions?: UiActions;
  taskRows: Task[];
  runtime: DirectorRuntime;
}) {
  const decisions = directorDecisionRows(taskRows, []);
  const suggestions = directorProposalRows(taskRows, fallbackSuggestionRows, runtime.sourceLabel);
  const blockedReasonMessages = directorBlockedReasonMessages(runtime.blockedReasons);
  const isBackendSource = runtime.sourceLabel.toLowerCase().includes('backend');
  const noDecisionsMessage = isBackendSource
    ? 'Backend queue has no recent decisions.'
    : 'No local decision rows are available.';
  const noSuggestionsMessage = isBackendSource
    ? 'No active backend proposals require review.'
    : 'No fallback suggestions are available.';

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
              <span className="rounded border border-line bg-row px-1.5 py-0.5 text-[9px] font-bold uppercase text-muted">{runtime.sourceLabel}</span>
            </div>
          </div>
        </div>
      </Panel>
      <Panel title="Current Goal">
        <p className="text-[11px] leading-5 text-muted">{runtime.currentGoal}</p>
        <button className="am-button mt-3" onClick={() => actions?.pushToast('Goal details are read-only in this cockpit view.', 'blue')}>Goal Details</button>
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
        {blockedReasonMessages.length > 0 && (
          <div className="mt-2 rounded-[5px] border border-warn/35 bg-warn/10 px-2 py-1.5 text-[10px] text-warn">
            {blockedReasonMessages.slice(0, 2).join(' · ')}
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
          {decisions.slice(0, 5).map(decision => (
            <div className="grid grid-cols-[38px_1fr_52px] gap-2 border-b border-line pb-2 text-[10px]" key={`${decision.time}-${decision.title}`}>
              <span className="font-mono text-muted">{decision.time}</span>
              <span className="truncate text-text">{decision.title}</span>
              <span className={`text-right ${toneText(decision.tone)}`}>{decision.status}</span>
            </div>
          ))}
          {!decisions.length && (
            <div className="rounded-[5px] border border-line bg-row px-2 py-2 text-[10px] leading-4 text-muted">
              {noDecisionsMessage}
            </div>
          )}
        </div>
      </Panel>
      <Panel title="Agent Suggestions">
        <div className="space-y-3 text-[10px]">
          {suggestions.map(suggestion => (
            <div key={suggestion.id}>
              <div className="flex items-center justify-between gap-2">
                <div className="truncate text-text">{suggestion.title}</div>
                <span className={`shrink-0 font-bold ${toneText(suggestion.tone)}`}>{suggestion.status}</span>
              </div>
              <div className="truncate text-muted">{suggestion.agent} - {suggestion.detail}</div>
              <div className="mt-2 flex gap-2">
                <button className="am-button text-online" onClick={() => actions?.pushToast(suggestion.reviewMessage, 'green')}>Review</button>
                <button className="am-button text-danger" onClick={() => actions?.pushToast(suggestion.rejectMessage, 'red')}>Reject</button>
              </div>
            </div>
          ))}
          {!suggestions.length && (
            <div className="rounded-[5px] border border-line bg-row px-2 py-2 leading-4 text-muted">
              {noSuggestionsMessage}
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
}
