import type { Tone } from '../types';

export type DashboardTab = 'Channels' | 'Instrument Space' | 'Overview' | 'EQ Match' | 'Dynamics' | 'Sends' | 'Main Bus';

export interface DashboardMetricSpec {
  id: string;
  label: string;
  accent: Tone;
  caption: string;
}

export interface ChannelDashboardTelemetry {
  waveform: number[];
  meterL: number;
  meterR: number;
  gateLabel: string;
  compLabel: string;
  safeState: string;
  pan: number;
  muted: boolean;
  solo: boolean;
  accent: Tone;
}

export interface AnalysisCardMock {
  id: string;
  title: string;
  accent: Tone;
  values: number[];
  primary: string;
  secondary: string;
  sourceLabel?: string;
  sourceTone?: Tone;
}

export interface DirectorDecisionMock {
  time: string;
  title: string;
  target: string;
  status: string;
  tone: Tone;
}

export interface AgentSuggestionMock {
  id: string;
  agent: string;
  title: string;
  delta: string;
  risk: string;
  confidence: number;
  tone: Tone;
}

export interface DirectorEventMock {
  time: string;
  source: string;
  message: string;
  tone: Tone;
}

export interface DashboardMockSafetyContract {
  source: 'frontend_dashboard_mock';
  sourceLabel: 'Display-only';
  sourceTone: Tone;
  caption: string;
  readOnly: true;
  liveWrite: false;
  canApplyToConsole: false;
  runtimeAuthority: false;
}

export const dashboardMockSafetyContract: DashboardMockSafetyContract = {
  source: 'frontend_dashboard_mock',
  sourceLabel: 'Display-only',
  sourceTone: 'yellow',
  caption: 'Display-only analyzer trace',
  readOnly: true,
  liveWrite: false,
  canApplyToConsole: false,
  runtimeAuthority: false
};

export const dashboardTabs: DashboardTab[] = [
  'Channels',
  'Instrument Space',
  'Overview',
  'EQ Match',
  'Dynamics',
  'Sends',
  'Main Bus'
];

export const analyticsCards: DashboardMetricSpec[] = [
  { id: 'master', label: 'Master Bus', accent: 'green', caption: 'Main L/R safety window' },
  { id: 'spectrum', label: 'Spectrum', accent: 'blue', caption: 'Analyzer FFT preview' },
  { id: 'loudness', label: 'Loudness', accent: 'yellow', caption: 'BS.1770 readout source-labeled in UI' },
  { id: 'balance', label: 'Mix Balance', accent: 'purple', caption: 'Source group distribution' },
  { id: 'confidence', label: 'AI Confidence', accent: 'green', caption: 'Director consensus' }
];

// Runtime dashboard DTO spectrum, phase, dynamics, and LUFS traces override
// these visual fallbacks when present.
export const spectrumBins = [28, 42, 36, 58, 76, 64, 48, 82, 70, 55, 43, 61, 78, 66, 51, 34, 47, 72, 86, 62, 40, 30];

export const mixBalanceSegments = [
  { label: 'Drums', value: 31, color: '#1687ff' },
  { label: 'Bass', value: 14, color: '#00d26a' },
  { label: 'Guitars', value: 18, color: '#ff9f1a' },
  { label: 'Vocals', value: 24, color: '#a855f7' },
  { label: 'FX', value: 13, color: '#ff4d4f' }
];

// Runtime channel snapshots override meter/mute/safety, pan, gate, and comp labels when present.
// Waveform stays a display-only fallback until analyzer frames have a backend DTO.
export const channelTelemetryById: Record<number, ChannelDashboardTelemetry> = {
  1: { waveform: [31, 66, 42, 78, 46, 84, 39, 72], meterL: -12.4, meterR: -13.2, gateLabel: 'G -42', compLabel: 'C 3.1', safeState: 'OK', pan: 0, muted: false, solo: false, accent: 'green' },
  2: { waveform: [28, 55, 39, 70, 43, 75, 36, 64], meterL: -14.1, meterR: -14.7, gateLabel: 'G -44', compLabel: 'C 2.8', safeState: 'OK', pan: 0, muted: false, solo: false, accent: 'green' },
  3: { waveform: [44, 82, 50, 88, 52, 91, 43, 80], meterL: -11.8, meterR: -12.4, gateLabel: 'G -38', compLabel: 'C 4.2', safeState: 'OK', pan: 0, muted: false, solo: false, accent: 'green' },
  4: { waveform: [36, 68, 42, 72, 45, 76, 39, 70], meterL: -13.6, meterR: -14.2, gateLabel: 'G -40', compLabel: 'C 3.6', safeState: 'OK', pan: 0, muted: false, solo: false, accent: 'green' },
  5: { waveform: [22, 46, 32, 54, 38, 59, 30, 48], meterL: -16.3, meterR: -17.1, gateLabel: 'G -50', compLabel: 'C 1.8', safeState: 'LOW', pan: 18, muted: false, solo: false, accent: 'blue' },
  6: { waveform: [34, 63, 41, 69, 44, 74, 36, 66], meterL: -13.0, meterR: -13.8, gateLabel: 'G -39', compLabel: 'C 2.9', safeState: 'OK', pan: -12, muted: false, solo: false, accent: 'green' },
  7: { waveform: [31, 60, 40, 66, 43, 70, 35, 62], meterL: -14.2, meterR: -14.6, gateLabel: 'G -41', compLabel: 'C 2.7', safeState: 'OK', pan: 14, muted: false, solo: false, accent: 'green' },
  8: { waveform: [18, 38, 45, 55, 49, 68, 42, 58], meterL: -18.1, meterR: -17.6, gateLabel: 'G OFF', compLabel: 'C 1.4', safeState: 'OK', pan: -38, muted: false, solo: false, accent: 'blue' },
  9: { waveform: [20, 40, 47, 58, 50, 70, 44, 60], meterL: -18.0, meterR: -17.5, gateLabel: 'G OFF', compLabel: 'C 1.4', safeState: 'OK', pan: 38, muted: false, solo: false, accent: 'blue' },
  10: { waveform: [48, 70, 52, 76, 55, 80, 50, 73], meterL: -10.5, meterR: -11.0, gateLabel: 'G -46', compLabel: 'C 5.0', safeState: 'OK', pan: 0, muted: false, solo: false, accent: 'yellow' },
  11: { waveform: [26, 52, 36, 60, 41, 65, 35, 56], meterL: -14.9, meterR: -15.2, gateLabel: 'G -48', compLabel: 'C 3.0', safeState: 'OK', pan: 0, muted: false, solo: false, accent: 'yellow' },
  12: { waveform: [55, 78, 68, 86, 73, 92, 64, 84], meterL: -9.3, meterR: -9.9, gateLabel: 'G -52', compLabel: 'C 4.6', safeState: 'FOCUS', pan: 0, muted: false, solo: true, accent: 'purple' },
  13: { waveform: [24, 45, 34, 58, 38, 62, 31, 50], meterL: -16.5, meterR: -17.0, gateLabel: 'G -54', compLabel: 'C 2.1', safeState: 'OK', pan: -20, muted: false, solo: false, accent: 'purple' },
  14: { waveform: [18, 36, 27, 45, 30, 48, 24, 40], meterL: -36.0, meterR: -36.0, gateLabel: 'G -54', compLabel: 'C 0.0', safeState: 'MUTE', pan: 20, muted: true, solo: false, accent: 'red' }
};

export const analysisCards: AnalysisCardMock[] = [
  { id: 'phase', title: 'Phase Scope', accent: 'green', values: [35, 55, 72, 48, 66, 82, 54, 70], primary: 'L/R stable', secondary: 'Lead ref aligned' },
  { id: 'correlation', title: 'Correlation', accent: 'blue', values: [52, 58, 61, 56, 64, 68, 63, 70], primary: '+0.28', secondary: 'Mono safe' },
  { id: 'spectral', title: 'Spectral Balance', accent: 'purple', values: [36, 58, 72, 64, 82, 49, 68, 56], primary: 'Vocal +1.2k', secondary: 'Masking low-mid' },
  { id: 'dynamics', title: 'Dynamics Overview', accent: 'yellow', values: [64, 52, 44, 70, 58, 48, 62, 55], primary: 'GR avg 3.1', secondary: 'Drums transient OK' },
  { id: 'loudness', title: 'Loudness History', accent: 'green', values: [61, 66, 64, 70, 68, 72, 69, 74, 71, 76], primary: '-14.2 LUFS', secondary: 'Target corridor' },
  { id: 'suggestions', title: 'AI Suggestions Over Time', accent: 'purple', values: [20, 48, 36, 72, 54, 68, 42, 80, 58, 70], primary: '23 pending', secondary: '3 blocked safely' }
];

// Mock-only fallback rows. Backend task/proposal/event rows render through
// directorPanelModel when the source is a Backend DTO.
export const directorDecisions: DirectorDecisionMock[] = [
  { time: '10:30', title: 'Lead Vocal held in front', target: 'CH 12', status: 'Review', tone: 'purple' },
  { time: '10:28', title: 'Snare transient verified', target: 'CH 03', status: 'Safe', tone: 'green' },
  { time: '10:27', title: 'Drum bus balance queued', target: 'BUS 01', status: 'Pending', tone: 'yellow' },
  { time: '10:26', title: 'Guitar resonance blocked', target: 'CH 15', status: 'Blocked', tone: 'red' }
];

export const agentSuggestions: AgentSuggestionMock[] = [
  { id: 'S-241', agent: 'Decision', title: 'Lower vocal harsh edge', delta: '-1.2 dB @ 4.1 kHz', risk: 'Low', confidence: 91, tone: 'purple' },
  { id: 'S-242', agent: 'Safety', title: 'Hold main bus ceiling', delta: 'limit margin 1.8 dB', risk: 'Guard', confidence: 96, tone: 'green' },
  { id: 'S-243', agent: 'Spectral', title: 'Clean guitar low-mid', delta: '-2.0 dB @ 260 Hz', risk: 'Medium', confidence: 84, tone: 'yellow' }
];

// Mock-only fallback rows. Backend sources render runtime-derived event rows in the Director panel.
export const directorEvents: DirectorEventMock[] = [
  { time: '10:31:08', source: 'Telemetry', message: 'Mock meter frame accepted for dashboard render.', tone: 'blue' },
  { time: '10:30:44', source: 'Director', message: 'No live OSC write issued from dashboard controls.', tone: 'green' },
  { time: '10:29:12', source: 'Analyzer', message: 'Lead Vocal remains selected for operator review.', tone: 'purple' },
  { time: '10:28:51', source: 'Safety', message: 'Safe limit and rollback controls remain visible.', tone: 'green' }
];

export const transportMock = {
  elapsed: '00:34:27',
  marker: 'Soundcheck Start',
  safeLimit: '-1.0 dBTP',
  stepSize: '0.5 dB',
  automation: 'Ride On'
};
