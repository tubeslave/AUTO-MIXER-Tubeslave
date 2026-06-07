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

export const modeDisplayLabel: Record<AutomixerMode, string> = {
  Analyze: 'Analysis Mode',
  Assist: 'Assisted Mixing',
  Supervised: 'Supervised AutoMix',
  Auto: 'Full Autonomous Mixing'
};

export function getModeDisplayLabel(mode: AutomixerMode): string {
  return modeDisplayLabel[mode];
}

const MODE_VALUE_ALIASES: Record<string, AutomixerMode> = {
  analyze: 'Analyze',
  analysis: 'Analyze',
  'analysis mode': 'Analyze',
  analysis_mode: 'Analyze',
  analysismode: 'Analyze',
  assist: 'Assist',
  assisted: 'Assist',
  'assisted mixing': 'Assist',
  assistedmixing: 'Assist',
  'assist mode': 'Assist',
  supervised: 'Supervised',
  'supervised automix': 'Supervised',
  'supervised automix mode': 'Supervised',
  'supervised auto mix': 'Supervised',
  supervised_auto_mix: 'Supervised',
  'supervised_auto_mode': 'Supervised',
  'supervised auto mode': 'Supervised',
  'supervised mode': 'Supervised',
  auto: 'Auto',
  autonomous: 'Auto',
  'full autonomous': 'Auto',
  'full autonomous mixing': 'Auto',
  'full autonomous mode': 'Auto',
  'full autonomous mixing mode': 'Auto',
  full_autonomous_mode: 'Auto',
  'auto mode': 'Auto',
};

const AI_AGENT_MODE_VALUE_ALIASES: Record<string, string> = {
  suggest: 'Suggest',
  suggestion: 'Suggest',
  'suggest mode': 'Suggest',
  suggest_mode: 'Suggest',
  manual: 'Manual',
  'manual mode': 'Manual',
  manual_mode: 'Manual',
  auto: 'Auto',
  autonomous: 'Auto',
  full_autonomous: 'Auto',
  'full_autonomous_mode': 'Auto',
  'full_autonomous_mixing': 'Auto',
  'full_autonomous_mixing_mode': 'Auto',
  auto_mode: 'Auto',
};

const RAW_MODE_MAP: Record<string, AutomixerMode> = {
  analyze: 'Analyze',
  assist: 'Assist',
  supervised: 'Supervised',
  auto: 'Auto',
};

export function formatAIAgentModeLabel(modeValue: unknown): string {
  const normalized = String(modeValue || '')
    .trim()
    .toLowerCase()
    .replace(/[-\s]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');

  return AI_AGENT_MODE_VALUE_ALIASES[normalized] || (normalized ? toTitleCase(normalized) : 'Suggest');
}

export function formatOperatorModeLabel(modeValue: unknown): string {
  const normalized = String(modeValue || '').trim().toLowerCase();
  const mode = MODE_VALUE_ALIASES[normalized] || RAW_MODE_MAP[normalized] || 'Assist';
  return getModeDisplayLabel(mode);
}

function toTitleCase(value: string): string {
  return value
    .split('_')
    .map(chunk => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(' ');
}

export function textClassForTone(tone?: Tone): string {
  if (tone === 'red') return 'text-danger';
  if (tone === 'yellow') return 'text-warn';
  if (tone === 'blue') return 'text-audio';
  if (tone === 'purple') return 'text-ai';
  if (tone === 'muted') return 'text-muted';
  return 'text-online';
}
