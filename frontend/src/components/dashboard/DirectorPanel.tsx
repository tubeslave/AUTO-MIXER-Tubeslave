import { DirectorAvatar, Panel, StatusBadge } from '../prototype/AutomixerComponents';
import {
  agentSuggestions,
  directorDecisions,
  directorEvents
} from '../../mock/dashboardMock';
import type { DirectorRuntime, Task, Tone, UiActions } from '../../types';

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
  const decisions = taskRows.length
    ? taskRows.slice(0, 4).map((task, index) => ({
      time: task.time,
      title: task.title,
      target: task.target ?? task.id,
      status: task.status,
      tone: directorDecisions[index]?.tone ?? 'blue'
    }))
    : directorDecisions;

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
            <p className="mt-1 text-[9px] text-online">● Online</p>
          </div>
        </div>
      </Panel>

      <Panel title="Текущая Цель" className="min-h-0 p-2.5">
        <p className="line-clamp-2 text-[10px] leading-4 text-muted">{runtime.currentGoal}</p>
        <button className="am-button mt-2 px-2 py-1" type="button">Детали цели</button>
      </Panel>

      <Panel title="Последние Решения" className="min-h-0 p-2.5">
        <div className="space-y-1">
          {decisions.map((decision) => (
            <div className="grid grid-cols-[38px_1fr_48px] items-center gap-2 border-b border-line/80 py-1 text-[9px]" key={`${decision.time}-${decision.title}`}>
              <span className="font-mono text-dim">{decision.time}</span>
              <span className="truncate font-semibold text-text">{decision.title}</span>
              <span className="truncate text-right font-bold text-online">Applied</span>
            </div>
          ))}
        </div>
        <button className="am-button mt-2 px-2 py-1" type="button">Все решения</button>
      </Panel>

      <Panel title="Предложения Агента" className="min-h-0 p-2.5">
        <div className="space-y-1.5">
          {agentSuggestions.slice(0, 2).map((suggestion) => (
            <div className="rounded-[4px] border border-line bg-row p-1.5 text-[9px]" key={suggestion.id}>
              <div className="flex items-center justify-between gap-2">
                <span className={toneText(suggestion.tone)}>{suggestion.agent}</span>
                <span className="font-mono text-muted">{suggestion.confidence}%</span>
              </div>
              <div className="mt-1 truncate font-bold text-text">{suggestion.title}</div>
              <div className="truncate text-muted">{suggestion.delta} · {suggestion.risk}</div>
              <div className="mt-1 grid grid-cols-2 gap-1.5">
                <button
                  className="am-button border-online/40 bg-online/10 px-2 py-0.5 text-[8px] text-online"
                  onClick={() => actions.pushToast(`${suggestion.id} accepted locally. No OSC write sent.`, 'green')}
                  type="button"
                >
                  Принять
                </button>
                <button
                  className="am-button border-danger/40 bg-danger/10 px-2 py-0.5 text-[8px] text-danger"
                  onClick={() => actions.pushToast(`${suggestion.id} rejected locally.`, 'red')}
                  type="button"
                >
                  Отклонить
                </button>
              </div>
            </div>
          ))}
        </div>
        <button className="am-button mt-2 px-2 py-1" type="button">Все предложения</button>
      </Panel>

      <Panel title="События" className="min-h-0 p-2.5">
        <div className="am-scrollbar h-full space-y-1.5 overflow-auto pr-1">
          {directorEvents.map((event) => (
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
