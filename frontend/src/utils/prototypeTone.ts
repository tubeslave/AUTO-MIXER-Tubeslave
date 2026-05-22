import type { AutomixerMode, Task, Tone } from '../types';

export function toneForPriority(priority: Task['priority'] | string): Tone {
  if (priority === 'High') return 'red';
  if (priority === 'Medium') return 'yellow';
  return 'green';
}

export function toneFromState(state: string): Tone {
  if (state === 'Locked') return 'purple';
  if (state === 'Review' || state === 'Running' || state === 'Queued') return 'yellow';
  if (state === 'Rejected' || state === 'Blocked') return 'red';
  if (state === 'Saved' || state === 'Complete' || state === 'Applied' || state === 'Active') return 'green';
  return 'blue';
}

export function toneForMode(mode: AutomixerMode): Tone {
  if (mode === 'Auto') return 'red';
  if (mode === 'Supervised') return 'yellow';
  if (mode === 'Assist') return 'purple';
  return 'blue';
}

export function textClassForTone(tone?: Tone): string {
  if (tone === 'red') return 'text-danger';
  if (tone === 'yellow') return 'text-warn';
  if (tone === 'blue') return 'text-audio';
  if (tone === 'purple') return 'text-ai';
  if (tone === 'muted') return 'text-muted';
  return 'text-online';
}
