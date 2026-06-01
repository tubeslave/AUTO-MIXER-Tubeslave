import { DirectorAvatar, Panel, StatusBadge } from '../prototype/AutomixerComponents';
import {
  agentSuggestions,
  directorDecisions,
  directorEvents
} from '../../mock/dashboardMock';
import type { DirectorRuntime, Task, Tone, UiActions } from '../../types';
import { directorDecisionRows, directorEventRows, directorProposalRows } from './directorPanelModel';

function toneText(tone: Tone) {
  if (tone === 'red') return 'text-danger';
  if (tone === 'yellow') return 'text-warn';
  if (tone === 'purple') return 'text-ai';
  if (tone === 'blue') return 'text-audio';
  return 'text-online';
}

export function AutomixerDirectorPanel({
  actions,
  runtime,
  taskRows
}: {
  actions: UiActions;
  runtime: DirectorRuntime;
  taskRows: Task[];
}) {
  const decisions = directorDecisionRows(taskRows, directorDecisions, runtime.sourceLabel);
  const events = directorEventRows(runtime, directorEvents);
  const proposalFallbackRows = agentSuggestions.slice(0, 2).map(suggestion => ({
    id: suggestion.id,
    agent: suggestion.agent,
    title: suggestion.title,
    detail: `${suggestion.delta} - ${suggestion.risk}`,
    status: 'Review',
    tone: suggestion.tone,
    reviewMessage: `${suggestion.id} reviewed locally. No OSC write sent.`,
    rejectMessage: `${suggestion.id} rejected locally. No OSC write sent.`
  }));
  const suggestions = directorProposalRows(taskRows, proposalFallbackRows, runtime.sourceLabel);
  const isBackendSource = runtime.sourceLabel.toLowerCase().includes('backend');
  const noDecisionsMessage = isBackendSource
    ? 'Backend queue has no recent decisions.'
    : 'Mock decision stream is unavailable.';
  const noSuggestionsMessage = isBackendSource
    ? 'No active backend proposals require review.'
    : 'No local fallback suggestions are available.';

  return (
    <div className="grid h-full min-h-0 grid-rows-[112px_94px_154px_130px_minmax(0,1fr)] gap-1.5">
      <Panel className="min-h-0 p-2.5">
        <div className="flex items-center gap-3">
          <DirectorAvatar />
          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-2">
              <h3 className="truncate text-[14px] font-black text-text">Automixer Director</h3>
              <StatusBadge tone="green">Online</StatusBadge>
            </div>
            <p className="mt-1 text-[10px] text-muted">Главный управляющий агент</p>
            <p className="mt-1 text-[9px] text-online">● Online · {runtime.sourceLabel}</p>
          </div>
        </div>
      </Panel>

      <Panel title="Текущая Цель" className="min-h-0 p-2.5">
        <p className="line-clamp-2 text-[10px] leading-4 text-muted">{runtime.currentGoal}</p>
        <button className="am-button mt-2 px-2 py-1" onClick={() => actions.pushToast('Goal details are read-only in this cockpit view.', 'blue')} type="button">Детали цели</button>
      </Panel>

      <Panel title="Последние Решения" className="min-h-0 p-2.5">
        <div className="space-y-1">
          {decisions.map((decision) => (
            <div className="grid grid-cols-[38px_1fr_48px] items-center gap-2 border-b border-line/80 py-1 text-[9px]" key={`${decision.time}-${decision.title}`}>
              <span className="font-mono text-dim">{decision.time}</span>
              <span className="truncate font-semibold text-text">{decision.title}</span>
              <span className={`truncate text-right font-bold ${toneText(decision.tone)}`}>{decision.status}</span>
            </div>
          ))}
          {!decisions.length && (
            <div className="rounded-[4px] border border-line bg-row px-2 py-2 text-[9px] text-muted">
              {noDecisionsMessage}
            </div>
          )}
        </div>
        <button className="am-button mt-2 px-2 py-1" onClick={() => actions.pushToast('Decision log opens as read-only evidence. No mixer write sent.', 'blue')} type="button">Все решения</button>
      </Panel>

      <Panel title="Предложения Агента" className="min-h-0 p-2.5">
        <div className="space-y-1.5">
          {suggestions.map((suggestion) => (
            <div className="rounded-[4px] border border-line bg-row p-1.5 text-[9px]" key={suggestion.id}>
              <div className="flex items-center justify-between gap-2">
                <span className={toneText(suggestion.tone)}>{suggestion.agent}</span>
                <span className={`font-bold ${toneText(suggestion.tone)}`}>{suggestion.status}</span>
              </div>
              <div className="mt-1 truncate font-bold text-text">{suggestion.title}</div>
              <div className="truncate text-muted">{suggestion.detail}</div>
              <div className="mt-1 grid grid-cols-2 gap-1.5">
                <button
                  className="am-button border-online/40 bg-online/10 px-2 py-0.5 text-[8px] text-online"
                  onClick={() => actions.pushToast(suggestion.reviewMessage, 'green')}
                  type="button"
                >
                  Просмотр
                </button>
                <button
                  className="am-button border-danger/40 bg-danger/10 px-2 py-0.5 text-[8px] text-danger"
                  onClick={() => actions.pushToast(suggestion.rejectMessage, 'red')}
                  type="button"
                >
                  Отклонить
                </button>
              </div>
            </div>
          ))}
          {!suggestions.length && (
            <div className="rounded-[4px] border border-line bg-row px-2 py-2 text-[9px] leading-4 text-muted">
              {noSuggestionsMessage}
            </div>
          )}
        </div>
        <button className="am-button mt-2 px-2 py-1" onClick={() => actions.pushToast('Agent suggestions are review-only here. Use Task Queue for supervised proposal flow.', 'blue')} type="button">Все предложения</button>
      </Panel>

      <Panel title="События" className="min-h-0 p-2.5">
        <div className="am-scrollbar h-full space-y-1.5 overflow-auto pr-1">
          {events.map((event) => (
            <div className="grid grid-cols-[42px_48px_1fr] gap-2 border-b border-line/80 pb-1.5 text-[9px]" key={`${event.time}-${event.message}`}>
              <span className="font-mono text-dim">{event.time}</span>
              <span className={toneText(event.tone)}>{event.source}</span>
              <span className="leading-4 text-muted">{event.message}</span>
            </div>
          ))}
        </div>
        <button className="am-button mt-2 px-2 py-1" type="button">Все события</button>
      </Panel>
    </div>
  );
}
