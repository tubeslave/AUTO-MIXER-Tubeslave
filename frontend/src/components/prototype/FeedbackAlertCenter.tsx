import type { FeedbackAlertRecord, Tone } from '../../types';
import { Panel, StatusBadge } from './AutomixerComponents';

const severityFrameClass: Record<Tone, string> = {
  red: 'border-danger/50 bg-danger/10 text-danger',
  yellow: 'border-warn/50 bg-warn/10 text-warn',
  blue: 'border-audio/50 bg-audio/10 text-audio',
  green: 'border-online/50 bg-online/10 text-online',
  purple: 'border-ai/50 bg-ai/10 text-ai',
  muted: 'border-line bg-panel2 text-muted'
};

export function FeedbackAlertBanner({ alerts }: { alerts: FeedbackAlertRecord[] }) {
  const latest = alerts[0] ?? null;

  if (!latest) {
    return (
      <Panel title="Feedback alert center">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge tone="muted">no feedback alerts</StatusBadge>
          <span className="text-[10px] text-muted">
            Live feedback monitor is detection-only (auto_action: none / read-only). Alerts from the backend appear here immediately.
          </span>
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="Feedback alert center" action={<StatusBadge tone={latest.severityTone}>{latest.severity}</StatusBadge>}>
      <div className={`rounded border p-2 ${severityFrameClass[latest.severityTone]}`}>
        <div className="flex flex-wrap items-end justify-between gap-2">
          <strong className="text-[16px] leading-none">
            FEEDBACK {latest.channelLabel} · {latest.frequencyLabel} · {latest.magnitudeLabel}
          </strong>
          <div className="flex flex-wrap items-center gap-1">
            <StatusBadge tone={latest.severityTone}>{latest.actionLabel}</StatusBadge>
            <StatusBadge tone="blue">confidence {latest.confidenceLabel}</StatusBadge>
            <StatusBadge tone="muted">{latest.timeLabel}</StatusBadge>
          </div>
        </div>
        <p className="mt-1 text-[9.5px] text-muted">
          Recommended action: {latest.actionLabel}. {latest.readOnlyNote} — the system detects and alerts only; suppression stays with the operator.
        </p>
      </div>
    </Panel>
  );
}

export function FeedbackAlertJournal({ alerts }: { alerts: FeedbackAlertRecord[] }) {
  return (
    <Panel title="Feedback journal">
      {alerts.length ? (
        <div className="am-scrollbar max-h-44 space-y-1 overflow-auto pr-1">
          {alerts.map((alert, index) => (
            <div
              className="flex items-center justify-between gap-2 rounded border border-line bg-panel2 px-2 py-1 text-[9.5px]"
              key={`${alert.timestamp}-${alert.channel}-${index}`}
            >
              <span className="min-w-0 truncate text-text">
                <strong>{alert.channelLabel}</strong> · {alert.frequencyLabel} · {alert.magnitudeLabel} · {alert.actionLabel}
              </span>
              <span className="flex shrink-0 items-center gap-1">
                <span className="text-muted">{alert.timeLabel}</span>
                <StatusBadge tone={alert.severityTone}>{alert.severity}</StatusBadge>
              </span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[10px] text-muted">Journal is empty. Newest backend feedback alerts list here (cap 50).</p>
      )}
    </Panel>
  );
}
