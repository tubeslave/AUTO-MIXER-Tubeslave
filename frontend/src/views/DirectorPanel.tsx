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

const decisionChipClass: Record<Tone, string> = {
  green: 'border-online/40 bg-online/12 text-online',
  blue: 'border-audio/40 bg-audio/12 text-audio',
  purple: 'border-ai/40 bg-ai/12 text-ai',
  yellow: 'border-warn/40 bg-warn/12 text-warn',
  red: 'border-danger/40 bg-danger/12 text-danger',
  muted: 'border-line2 bg-panel2 text-muted'
};

function decisionChipLabel(status: string) {
  const normalized = status.toLowerCase();
  if (normalized === 'applied') return 'APPLIED';
  if (normalized === 'blocked') return 'BLOCKED';
  if (normalized === 'review') return 'REVIEW';
  if (normalized === 'pending') return 'PENDING';
  if (normalized === 'safe') return 'SAFE';
  return status.toUpperCase();
}

function SectionAllButton({ label, onClick }: { label: string; onClick?: () => void }) {
  return (
    <button
      className="mt-2 w-full rounded-[5px] border border-line bg-panel2 px-2 py-1.5 text-center text-[9px] font-black uppercase tracking-[0.1em] text-muted transition hover:border-line2 hover:text-text"
      onClick={onClick}
      type="button"
    >
      {label}
    </button>
  );
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
  const suggestionBlockerNotice = blockedReasonMessages.length > 0 && suggestions.length > 0
    ? `Promotion planning blocked. Suggestions remain review-only: ${blockedReasonMessages[0]}`
    : '';

  return (
    <div className="flex flex-col gap-2">
      <Panel>
        <div className="flex items-center gap-3">
          <DirectorAvatar />
          <div className="min-w-0">
            <h3 className="text-[14px] font-bold leading-tight text-text">Automixer Director</h3>
            <p className="text-[9.5px] text-muted">Главный управляющий агент</p>
            <div className="mt-1.5 flex items-center gap-1.5">
              <StatusBadge tone="green">Online</StatusBadge>
              <span className="rounded border border-line bg-row px-1.5 py-0.5 text-[8px] font-bold uppercase text-muted">{runtime.sourceLabel}</span>
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="Текущая цель">
        <p className="line-clamp-4 text-[10.5px] leading-[17px] text-text/90">{runtime.currentGoal}</p>
        <SectionAllButton label="Детали цели" onClick={() => actions?.pushToast('Goal details are read-only in this cockpit view.', 'blue')} />
      </Panel>

      <Panel title="Runtime Guard">
        <div className="mb-2.5 grid grid-cols-2 gap-1.5 text-[10px]">
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

      <Panel title="Последние решения" className="flex-1">
        <div className="space-y-[7px]">
          {decisions.slice(0, 5).map(decision => (
            <div className="grid grid-cols-[32px_1fr_auto] items-center gap-2 border-b border-line/70 pb-[7px] text-[10px]" key={`${decision.time}-${decision.title}`}>
              <span className="font-mono text-[9px] text-dim">{decision.time}</span>
              <span className="truncate text-text/90">{decision.title}</span>
              <span className={`rounded-[3px] border px-1.5 py-[2px] text-[7px] font-black uppercase tracking-[0.06em] ${decisionChipClass[decision.tone]}`}>
                {decisionChipLabel(decision.status)}
              </span>
            </div>
          ))}
          {!decisions.length && (
            <div className="rounded-[5px] border border-line bg-row px-2 py-2 text-[10px] leading-4 text-muted">
              {noDecisionsMessage}
            </div>
          )}
        </div>
        <SectionAllButton label="Все решения" onClick={() => actions?.pushToast('Full decision log opens in Decision Log view.', 'blue')} />
      </Panel>

      <Panel title="Предложения агента">
        <div className="space-y-3 text-[10px]">
          {suggestionBlockerNotice && (
            <div className="rounded-[5px] border border-danger/35 bg-danger/10 px-2 py-1.5 leading-4 text-danger">
              {suggestionBlockerNotice}
            </div>
          )}
          {suggestions.map(suggestion => (
            <div key={suggestion.id}>
              <div className="flex items-center justify-between gap-2">
                <div className="truncate font-semibold text-text">{suggestion.title}</div>
                <span className={`shrink-0 text-[8px] font-black uppercase ${toneText(suggestion.tone)}`}>{suggestion.status}</span>
              </div>
              <div className="mt-0.5 truncate text-muted">{suggestion.agent} — {suggestion.detail}</div>
              <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                <button
                  className="rounded-[4px] border border-online/50 bg-online/14 px-2 py-1 text-[9px] font-black uppercase tracking-[0.06em] text-online transition hover:bg-online/25"
                  onClick={() => actions?.pushToast(suggestion.reviewMessage, 'green')}
                  type="button"
                >
                  Принять
                </button>
                <button
                  className="rounded-[4px] border border-danger/50 bg-danger/10 px-2 py-1 text-[9px] font-black uppercase tracking-[0.06em] text-danger transition hover:bg-danger/20"
                  onClick={() => actions?.pushToast(suggestion.rejectMessage, 'red')}
                  type="button"
                >
                  Отклонить
                </button>
              </div>
            </div>
          ))}
          {!suggestions.length && (
            <div className="rounded-[5px] border border-line bg-row px-2 py-2 leading-4 text-muted">
              {noSuggestionsMessage}
            </div>
          )}
        </div>
        <SectionAllButton label="Все предложения" onClick={() => actions?.pushToast('Full proposal queue opens in Decision Log view.', 'blue')} />
      </Panel>
    </div>
  );
}
