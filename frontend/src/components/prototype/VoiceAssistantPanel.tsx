import { VOICE_ASSISTANT_BINDING_NOTE, VOICE_ASSISTANT_LANGUAGES } from '../voiceAssistantPanel.mjs';
import type { Tone, UiActions, VoiceAssistantState } from '../../types';
import { Panel, StatusBadge } from './AutomixerComponents';

type VoiceAssistantPanelProps = {
  state: VoiceAssistantState;
  onSetLanguage: UiActions['setVoiceLanguage'];
  onToggleEnabled: UiActions['toggleVoiceEnabled'];
};

// Ф10 (DEC-0010): панель голосового помощника саундчека. Этап 0 backend —
// локальный TTS (macOS say), НИКАКИХ записей на пульт. WS-обвязки пока нет,
// поэтому переключатели меняют только локальное UI-состояние (см. note).
export function VoiceAssistantPanel({ state, onSetLanguage, onToggleEnabled }: VoiceAssistantPanelProps) {
  const enabledTone: Tone = state.enabled ? 'green' : 'muted';
  return (
    <Panel
      action={<StatusBadge tone={enabledTone}>{state.enabled ? 'voice on' : 'voice off'}</StatusBadge>}
      title="Voice assistant (Ф10)"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] text-muted">Language</span>
        {VOICE_ASSISTANT_LANGUAGES.map(lang => (
          <button
            className="am-button"
            disabled={state.lang === lang}
            key={lang}
            onClick={() => onSetLanguage(lang as VoiceAssistantState['lang'])}
            type="button"
          >
            {lang.toUpperCase()}
          </button>
        ))}
        <button className="am-button" onClick={onToggleEnabled} type="button">
          {state.enabled ? 'Mute voice' : 'Enable voice'}
        </button>
        <StatusBadge tone="blue">{state.lang}</StatusBadge>
      </div>
      <p className="mt-1 text-[9.5px] text-muted">{VOICE_ASSISTANT_BINDING_NOTE}</p>
      {state.spokenLog.length ? (
        <div className="am-scrollbar mt-2 max-h-32 space-y-1 overflow-auto pr-1">
          {state.spokenLog.map((entry, index) => (
            <div
              className="flex items-center justify-between gap-2 rounded border border-line bg-panel2 px-2 py-1 text-[9.5px]"
              key={`${entry.phrase}-${index}`}
            >
              <span className="min-w-0 truncate text-text">{entry.text}</span>
              <StatusBadge tone="muted">{entry.phrase}</StatusBadge>
            </div>
          ))}
        </div>
      ) : (
        <p className="mt-2 text-[10px] text-muted">
          Spoken log is empty. Phrases appear here once the backend push (spoken_log) is bound; cap 50.
        </p>
      )}
    </Panel>
  );
}
