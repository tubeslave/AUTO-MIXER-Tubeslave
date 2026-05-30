import type {
  Agent,
  ApplyPlanStep,
  AutomixerMode,
  BusRecord,
  Channel,
  Communication,
  DecisionPolicy,
  DeviceRecord,
  LogRecord,
  MemoryRecord,
  Metric,
  NavItem,
  PerformanceRecord,
  ProposalPreview,
  SafetyEvidenceItem,
  Task
} from '../types';

export const navItems: NavItem[] = [
  { id: 'dashboard', label: 'Dashboard', section: 'Overview', mark: 'OV' },
  { id: 'channels', label: 'Channel List', section: 'Channels', mark: 'CH' },
  { id: 'groups', label: 'Groups / Buses', section: 'Channels', mark: 'GB' },
  { id: 'agents', label: 'Agents Console', section: 'AI Agents', mark: 'AI' },
  { id: 'communications', label: 'Communications', section: 'AI Agents', mark: 'CM' },
  { id: 'task-queue', label: 'Task Queue', section: 'AI Agents', mark: 'TQ' },
  { id: 'memory', label: 'Memory / Knowledge Base', section: 'AI Agents', mark: 'MB' },
  { id: 'performance', label: 'Performance', section: 'AI Agents', mark: 'PF' },
  { id: 'training', label: 'Training Sessions', section: 'Learning', mark: 'TR' },
  { id: 'rules', label: 'Rules Library', section: 'Learning', mark: 'RL' },
  { id: 'system', label: 'System Connection', section: 'System', mark: 'SC' },
  { id: 'routing', label: 'Audio Routing', section: 'System', mark: 'AR' },
  { id: 'osc-midi', label: 'OSC / MIDI', section: 'System', mark: 'OM' },
  { id: 'devices', label: 'Devices', section: 'System', mark: 'DV' },
  { id: 'settings', label: 'Settings', section: 'System', mark: 'ST' },
  { id: 'logs', label: 'Logs', section: 'System', mark: 'LG' }
];

export const overviewMetrics: Metric[] = [
  { label: 'Active Agents', value: '8', sub: 'all systems working', tone: 'green' },
  { label: 'Decisions Today', value: '142', sub: '+28% from yesterday', tone: 'blue' },
  { label: 'AI Confidence', value: '87%', sub: 'high', tone: 'green' },
  { label: 'Suggestions Pending', value: '23', sub: 'requires review', tone: 'yellow' },
  { label: 'Actions Applied', value: '12', sub: 'last 30 min', tone: 'green' },
  { label: 'Warnings', value: '3', sub: 'needs attention', tone: 'yellow' }
];

export const channels: Channel[] = [
  { id: 1, name: 'Kick In', type: 'Kick', source: 'In 01', level: -12.4, lufs: -14.2, dynamic: 8.1, status: 'ACTIVE', recognition: 'Kick (98%)', group: 'DRUMS', fader: -5.4, confidence: 98 },
  { id: 2, name: 'Kick Out', type: 'Kick', source: 'In 02', level: -14.1, lufs: -16.3, dynamic: 7.8, status: 'ACTIVE', recognition: 'Kick (97%)', group: 'DRUMS', fader: -6.1, confidence: 97 },
  { id: 3, name: 'Snare Top', type: 'Snare', source: 'In 03', level: -11.8, lufs: -13.1, dynamic: 9.3, status: 'ACTIVE', recognition: 'Snare (99%)', group: 'DRUMS', fader: -7.3, confidence: 99 },
  { id: 4, name: 'Snare Bottom', type: 'Snare', source: 'In 04', level: -13.6, lufs: -15.4, dynamic: 8.6, status: 'ACTIVE', recognition: 'Snare (96%)', group: 'DRUMS', fader: -8.0, confidence: 96 },
  { id: 5, name: 'Hi-Hat', type: 'Hi-Hat', source: 'In 05', level: -16.3, lufs: -18.9, dynamic: 6.2, status: 'ACTIVE', recognition: 'Hi-Hat (95%)', group: 'DRUMS', fader: -14.2, confidence: 95 },
  { id: 6, name: 'Tom 1', type: 'Tom', source: 'In 06', level: -13.0, lufs: -15.7, dynamic: 8.4, status: 'ACTIVE', recognition: 'Tom (94%)', group: 'DRUMS', fader: -10.6, confidence: 94 },
  { id: 7, name: 'Tom 2', type: 'Tom', source: 'In 07', level: -14.2, lufs: -16.0, dynamic: 8.0, status: 'ACTIVE', recognition: 'Tom (94%)', group: 'DRUMS', fader: -11.3, confidence: 94 },
  { id: 8, name: 'OH L', type: 'Overhead', source: 'In 08', level: -18.1, lufs: -20.3, dynamic: 10.2, status: 'ACTIVE', recognition: 'Overhead (98%)', group: 'DRUMS', fader: -8.9, confidence: 98 },
  { id: 9, name: 'OH R', type: 'Overhead', source: 'In 09', level: -18.0, lufs: -20.1, dynamic: 10.5, status: 'ACTIVE', recognition: 'Overhead (98%)', group: 'DRUMS', fader: -8.7, confidence: 98 },
  { id: 10, name: 'Bass DI', type: 'Bass', source: 'In 10', level: -10.5, lufs: -11.8, dynamic: 7.1, status: 'ACTIVE', recognition: 'Bass (99%)', group: 'BASS', fader: -6.2, confidence: 99 },
  { id: 11, name: 'Bass Amp', type: 'Bass', source: 'In 11', level: -14.9, lufs: -16.8, dynamic: 6.9, status: 'ACTIVE', recognition: 'Bass (97%)', group: 'BASS', fader: -10.1, confidence: 97 },
  { id: 12, name: 'Lead Vocal', type: 'Vocal Lead', source: 'In 12', level: -9.3, lufs: -10.2, dynamic: 6.3, status: 'ACTIVE', recognition: 'Lead Vocal (99%)', group: 'VOCALS', fader: -12.3, confidence: 99 },
  { id: 13, name: 'Backing Vocal 1', type: 'Vocal Back', source: 'In 13', level: -16.5, lufs: -17.9, dynamic: 5.4, status: 'ACTIVE', recognition: 'Backing Vocal (96%)', group: 'VOCALS', fader: -15.8, confidence: 96 },
  { id: 14, name: 'Backing Vocal 2', type: 'Vocal Back', source: 'In 14', level: -16.7, lufs: -18.0, dynamic: 5.6, status: 'MUTED', recognition: 'Backing Vocal (95%)', group: 'VOCALS', fader: -50.0, confidence: 95 },
  { id: 15, name: 'Gtr 1', type: 'Guitar Elec', source: 'In 15', level: -12.7, lufs: -14.6, dynamic: 7.3, status: 'ACTIVE', recognition: 'Electric Guitar (97%)', group: 'GUITARS', fader: -9.3, confidence: 97 },
  { id: 16, name: 'Gtr 2', type: 'Guitar Elec', source: 'In 16', level: -13.1, lufs: -15.1, dynamic: 7.6, status: 'ACTIVE', recognition: 'Electric Guitar (96%)', group: 'GUITARS', fader: -9.7, confidence: 96 }
];

export const agents: Agent[] = [
  { id: 'analyzer', name: 'Analyzer Agent', role: 'Audio Analysis', status: 'ACTIVE', task: 'Analyzing 14 channels', confidence: 92, activity: [20, 62, 35, 76, 44, 84, 52, 70, 48, 86], lastAction: 'Balance matched Lead Vocal', tone: 'green' },
  { id: 'decision', name: 'Decision Agent', role: 'Mix Decisions', status: 'ACTIVE', task: 'Evaluating mix balance', confidence: 87, activity: [34, 46, 62, 40, 74, 42, 58, 82, 36, 66], lastAction: 'Suggested vocal -1.5 dB', tone: 'purple' },
  { id: 'critic', name: 'Mix Critic Agent', role: 'Quality Control', status: 'ACTIVE', task: 'Checking confirmation', confidence: 89, activity: [42, 82, 55, 72, 30, 66, 90, 62, 80, 46], lastAction: 'Approved improvement', tone: 'blue' },
  { id: 'spectral', name: 'Spectral Agent', role: 'Spectrum Balance', status: 'ACTIVE', task: 'Monitoring spectrum', confidence: 81, activity: [22, 44, 38, 88, 54, 76, 42, 68, 30, 80], lastAction: 'Created HPF proposal', tone: 'yellow' },
  { id: 'safety', name: 'Safety Agent', role: 'Safety & Limits', status: 'ACTIVE', task: 'Verifying all changes', confidence: 95, activity: [56, 70, 48, 92, 62, 84, 52, 76, 68, 90], lastAction: 'All changes safe', tone: 'green' },
  { id: 'osc', name: 'OSC Controller Agent', role: 'Console Commands', status: 'STANDBY', task: 'Waiting for approval', confidence: 100, activity: [18, 18, 22, 20, 28, 24, 18, 26, 22, 20], lastAction: 'Command prepared', tone: 'blue' },
  { id: 'learning', name: 'Learning Agent', role: 'Rule Learning', status: 'WARNING', task: 'Creating new rule', confidence: 76, activity: [18, 54, 26, 62, 30, 58, 38, 68, 34, 52], lastAction: 'New rule created', tone: 'yellow' },
  { id: 'memory', name: 'Memory Agent', role: 'Knowledge Sync', status: 'ACTIVE', task: 'Syncing database', confidence: 90, activity: [40, 66, 44, 72, 60, 80, 50, 74, 58, 86], lastAction: 'Knowledge base updated', tone: 'green' }
];

export const tasks: Task[] = [
  { id: 'T-101', priority: 'High', title: 'Reduce Lead Vocal presence', agent: 'Decision Agent', status: 'Pending', time: '10:30', target: 'Lead Vocal' },
  { id: 'T-102', priority: 'High', title: 'Review Snare transient', agent: 'Mix Critic Agent', status: 'Review', time: '10:28', target: 'Snare Top' },
  { id: 'T-103', priority: 'Medium', title: 'Balance drum bus', agent: 'Analyzer Agent', status: 'Pending', time: '10:27', target: 'DRUMS' },
  { id: 'T-104', priority: 'Medium', title: 'Check guitar resonances', agent: 'Spectral Agent', status: 'Pending', time: '10:26', target: 'GUITARS' },
  { id: 'T-105', priority: 'Low', title: 'Update vocal EQ rule', agent: 'Learning Agent', status: 'Applied', time: '10:24', target: 'VOCALS' },
  { id: 'T-106', priority: 'Low', title: 'Sync recent decisions', agent: 'Memory Agent', status: 'Applied', time: '10:23', target: 'Memory' },
  { id: 'T-107', priority: 'Medium', title: 'Verify vocal send to FX15', agent: 'Safety Agent', status: 'Review', time: '10:21', target: 'FX Returns' },
  { id: 'T-108', priority: 'High', title: 'Hold guitar bus under vocal', agent: 'Decision Agent', status: 'Pending', time: '10:20', target: 'GUITARS' }
];

export function decisionPolicyForMode(mode: AutomixerMode): DecisionPolicy {
  if (mode === 'Analyze') {
    return {
      actionLabel: 'Analyze',
      apply: 'Blocked',
      write: 'Disabled',
      rollback: 'Not needed',
      hint: 'Read-only mode. The director can inspect the proposal, but cannot queue or apply it.',
      chain: [
        { node: 'Analyzer Agent', state: 'Signal read', tone: 'blue' },
        { node: 'Safety Agent', state: 'Read-only', tone: 'blue' },
        { node: 'Director', state: 'Blocked', tone: 'red' },
        { node: 'Write Path', state: 'Disabled', tone: 'red' }
      ]
    };
  }

  if (mode === 'Assist') {
    return {
      actionLabel: 'Queue Review',
      apply: 'Review only',
      write: 'Disabled',
      rollback: 'Local',
      hint: 'Assist mode can propose and prepare operator review, but it cannot apply mixer changes.',
      chain: [
        { node: 'Analyzer Agent', state: 'Suggested', tone: 'purple' },
        { node: 'Safety Agent', state: 'Ready', tone: 'yellow' },
        { node: 'Director', state: 'Review', tone: 'yellow' },
        { node: 'Write Path', state: 'Disabled', tone: 'red' }
      ]
    };
  }

  if (mode === 'Supervised') {
    return {
      actionLabel: 'Approve',
      apply: 'Backend gate',
      write: 'Supervised',
      rollback: 'Backend',
      hint: 'Supervised mode can request backend apply only through the operator proposal gate. Direct OSC writes remain unavailable from the UI.',
      chain: [
        { node: 'Analyzer Agent', state: 'Ready', tone: 'green' },
        { node: 'Safety Agent', state: 'Required', tone: 'yellow' },
        { node: 'Operator Gate', state: 'Approve', tone: 'green' },
        { node: 'Write Path', state: 'Guarded', tone: 'yellow' }
      ]
    };
  }

  return {
    actionLabel: 'Sim Apply',
    apply: 'Simulated',
    write: 'Disabled',
    rollback: 'Local',
    hint: 'Auto mode is a visual simulation only in this prototype. It intentionally does not send live commands.',
    chain: [
      { node: 'Analyzer Agent', state: 'Running', tone: 'green' },
      { node: 'Director', state: 'Auto sim', tone: 'red' },
      { node: 'Safety Agent', state: 'Guarded', tone: 'yellow' },
      { node: 'Write Path', state: 'Disabled', tone: 'red' }
    ]
  };
}

export function proposalPreviewForTask(task: Task): ProposalPreview {
  if (task.reviewOnly) {
    const contract = task.proposalContract;
    const valueType = task.requestedValueType?.replace(/_/g, ' ') || 'review';
    const evidence = contract?.safetyEvidenceRequired?.slice(0, 2).map(item => item.replace(/_/g, ' ')).join(' + ');
    return {
      parameter: valueType,
      delta: 'Review only',
      risk: 'High',
      proof: evidence || 'Hardware proof missing',
      rollback: contract?.rollbackRequired ? 'Rollback required' : 'Not approved',
      tone: 'red'
    };
  }

  if (task.title.includes('Lead Vocal')) {
    return {
      parameter: 'Vocal fader',
      delta: '-1.5 dB',
      risk: 'Medium',
      proof: 'Headroom + intelligibility',
      rollback: 'Restore fader snapshot',
      tone: 'yellow'
    };
  }

  if (task.title.includes('Snare')) {
    return {
      parameter: 'HPF',
      delta: '120 Hz',
      risk: 'Low',
      proof: 'Transient check',
      rollback: 'Restore EQ band',
      tone: 'green'
    };
  }

  if (task.title.includes('drum')) {
    return {
      parameter: 'Bus balance',
      delta: '+0.8 dB',
      risk: 'Medium',
      proof: 'Drum bus headroom',
      rollback: 'Restore bus trim',
      tone: 'yellow'
    };
  }

  if (task.title.includes('guitar')) {
    return {
      parameter: 'Resonance EQ',
      delta: '-2 dB @ 3.2 kHz',
      risk: 'High',
      proof: 'Vocal masking check',
      rollback: 'Bypass EQ move',
      tone: 'red'
    };
  }

  if (task.title.includes('EQ rule')) {
    return {
      parameter: 'Rule memory',
      delta: 'Save draft',
      risk: 'Low',
      proof: 'No live write',
      rollback: 'Discard draft',
      tone: 'green'
    };
  }

  if (task.title.includes('FX15')) {
    return {
      parameter: 'FX send',
      delta: '-1 dB',
      risk: 'Medium',
      proof: 'Return level check',
      rollback: 'Restore send',
      tone: 'yellow'
    };
  }

  return {
    parameter: 'Decision state',
    delta: 'Sync',
    risk: 'Low',
    proof: 'Local audit',
    rollback: 'Clear override',
    tone: 'blue'
  };
}

export function safetyEvidenceForTask(task: Task, mode: AutomixerMode): SafetyEvidenceItem[] {
  if (task.safetyEvidence?.length) {
    const ensemble = task.safetyEvidence.filter(row => row.label.toLowerCase().includes('ensemble'));
    const remaining = task.safetyEvidence.filter(row => !row.label.toLowerCase().includes('ensemble'));
    return [...ensemble, ...remaining].slice(0, 6);
  }

  if (task.reviewOnly) {
    const contract = task.proposalContract;
    const evidence = contract?.safetyEvidenceRequired ?? [];
    return [
      { label: 'Live apply', value: 'Blocked', tone: 'red' },
      { label: 'Value type', value: task.requestedValueType?.replace(/_/g, ' ') || 'Unsupported', tone: 'red' },
      { label: 'Readback', value: contract?.readbackRequired ? 'Required' : 'Missing', tone: 'yellow' },
      { label: 'Rollback', value: contract?.rollbackRequired ? 'Required' : 'Missing', tone: 'yellow' },
      { label: 'Evidence', value: evidence[0]?.replace(/_/g, ' ') || 'Required', tone: 'yellow' },
      { label: 'Promotion', value: 'Not proven', tone: 'red' }
    ];
  }

  const proposal = proposalPreviewForTask(task);
  const highRisk = proposal.risk === 'High' || task.priority === 'High';
  const canOnlyReview = mode === 'Analyze' || mode === 'Assist';

  return [
    { label: 'Headroom', value: highRisk ? 'Required' : 'Ready', tone: highRisk ? 'yellow' : 'green' },
    { label: 'Peak', value: '-1.1 dBTP', tone: 'green' },
    { label: 'Feedback', value: highRisk ? 'Review' : 'Clear', tone: highRisk ? 'yellow' : 'green' },
    { label: 'Readback', value: canOnlyReview ? 'Pending' : 'Planned', tone: canOnlyReview ? 'yellow' : 'blue' },
    { label: 'Snap', value: 'Required', tone: 'yellow' },
    { label: 'Undo', value: proposal.rollback, tone: 'blue' }
  ];
}

export function ensembleEvidenceForTask(task: Task): SafetyEvidenceItem[] {
  if (!task.safetyEvidence?.length) return [];

  const preferredLabels = [
    'ensemble context',
    'ensemble time coherence',
    'ensemble time sources',
    'ensemble lag samples',
    'ensemble apply policy'
  ];
  const evidenceByLabel = new Map(
    task.safetyEvidence.map(row => [row.label.toLowerCase(), row])
  );

  return preferredLabels
    .map(label => evidenceByLabel.get(label))
    .filter((row): row is SafetyEvidenceItem => Boolean(row));
}

export function applyPlanForTask(task: Task, mode: AutomixerMode): ApplyPlanStep[] {
  if (task.reviewOnly) {
    return [
      { label: 'Snap', value: 'N/A', tone: 'muted' },
      { label: 'Gate', value: 'Review', tone: 'yellow' },
      { label: 'Apply', value: 'Blocked', tone: 'red' },
      { label: 'Read', value: 'Required', tone: 'yellow' },
      { label: 'Undo', value: 'No live', tone: 'red' }
    ];
  }

  const proposal = proposalPreviewForTask(task);
  const snapshotState = proposal.risk === 'Low' ? 'Ready' : 'Required';

  if (mode === 'Analyze') {
    return [
      { label: 'Snap', value: 'N/A', tone: 'muted' },
      { label: 'Gate', value: 'Blocked', tone: 'red' },
      { label: 'Apply', value: 'Off', tone: 'red' },
      { label: 'Read', value: 'N/A', tone: 'muted' },
      { label: 'Undo', value: 'N/A', tone: 'muted' }
    ];
  }

  if (mode === 'Assist') {
    return [
      { label: 'Snap', value: snapshotState, tone: snapshotState === 'Ready' ? 'green' : 'yellow' },
      { label: 'Gate', value: 'Review', tone: 'yellow' },
      { label: 'Apply', value: 'Hold', tone: 'red' },
      { label: 'Read', value: 'Pending', tone: 'yellow' },
      { label: 'Undo', value: 'Draft', tone: 'blue' }
    ];
  }

  if (mode === 'Supervised') {
    return [
      { label: 'Snap', value: snapshotState, tone: snapshotState === 'Ready' ? 'green' : 'yellow' },
      { label: 'Gate', value: 'Operator', tone: 'green' },
      { label: 'Apply', value: 'Backend', tone: 'yellow' },
      { label: 'Read', value: 'Planned', tone: 'yellow' },
      { label: 'Undo', value: 'Ready', tone: 'green' }
    ];
  }

  return [
    { label: 'Snap', value: snapshotState, tone: snapshotState === 'Ready' ? 'green' : 'yellow' },
    { label: 'Gate', value: 'Sim', tone: 'red' },
    { label: 'Apply', value: 'Mock', tone: 'yellow' },
    { label: 'Read', value: 'Mock', tone: 'blue' },
    { label: 'Undo', value: 'Local', tone: 'blue' }
  ];
}

export const communications: Communication[] = [
  { time: '10:30', from: 'Analyzer Agent', to: 'Decision Agent', topic: 'Vocal energy exceeds reference by 1.5 dB', status: 'New', tone: 'purple' },
  { time: '10:29', from: 'Safety Agent', to: 'Decision Agent', topic: 'Headroom available, no feedback risk', status: 'Approved', tone: 'green' },
  { time: '10:28', from: 'Mix Critic Agent', to: 'Analyzer Agent', topic: 'Transient check requested for snare', status: 'Review', tone: 'yellow' },
  { time: '10:27', from: 'Learning Agent', to: 'Memory Agent', topic: 'Store successful vocal balance rule', status: 'Saved', tone: 'blue' },
  { time: '10:26', from: 'OSC Controller', to: 'Director', topic: 'No write sent, waiting for supervised approval', status: 'Blocked', tone: 'red' }
];

export const memories: MemoryRecord[] = [
  { name: 'Lead vocal presence rule', type: 'Mix Rule', size: '2.1 KB', confidence: 98, updated: '10:24' },
  { name: 'Snare HPF venue profile', type: 'EQ Knowledge', size: '4.7 KB', confidence: 92, updated: '10:18' },
  { name: 'Club wing routing map', type: 'Routing', size: '8.4 KB', confidence: 99, updated: '10:12' },
  { name: 'Bass DI compression memory', type: 'Dynamics', size: '3.2 KB', confidence: 87, updated: '10:05' },
  { name: 'Overhead phase pattern', type: 'Phase', size: '1.9 KB', confidence: 91, updated: '09:58' },
  { name: 'Vocal rides from last show', type: 'Automation', size: '5.5 KB', confidence: 94, updated: '09:42' },
  { name: 'Wing Rack supervised write gate', type: 'Safety Rule', size: '6.8 KB', confidence: 100, updated: '09:30' },
  { name: 'Drum bus transient profile', type: 'Dynamics', size: '4.1 KB', confidence: 89, updated: '09:22' }
];

export const performance: PerformanceRecord[] = [
  { agent: 'Analyzer Agent', cpu: '14%', memory: '928 MB', decisions: 329, success: '98%', latency: '12 ms', status: 'Good' },
  { agent: 'Decision Agent', cpu: '18%', memory: '640 MB', decisions: 147, success: '96%', latency: '18 ms', status: 'Good' },
  { agent: 'Mix Critic Agent', cpu: '22%', memory: '752 MB', decisions: 88, success: '91%', latency: '26 ms', status: 'Good' },
  { agent: 'Spectral Agent', cpu: '16%', memory: '584 MB', decisions: 104, success: '94%', latency: '16 ms', status: 'Good' },
  { agent: 'Learning Agent', cpu: '31%', memory: '1.2 GB', decisions: 24, success: '84%', latency: '109 ms', status: 'Watch' }
];

export const devices: DeviceRecord[] = [
  { name: 'WING RACK', ip: '192.168.1.10', type: 'Audio Console', sampleRate: '48 kHz', io: '32 Inputs / 16 Outputs', status: 'Online' },
  { name: 'Stage Box 1', ip: '192.168.1.11', type: 'I/O Expander', sampleRate: '48 kHz', io: '16 Inputs / 8 Outputs', status: 'Online' },
  { name: 'Stage Box 2', ip: '192.168.1.12', type: 'I/O Expander', sampleRate: '48 kHz', io: '16 Inputs / 8 Outputs', status: 'Online' },
  { name: 'Powerplay P16', ip: '192.168.1.13', type: 'Monitor System', sampleRate: '48 kHz', io: '16 Outputs', status: 'Online' },
  { name: 'Dante Converter', ip: '192.168.1.14', type: 'Dante Device', sampleRate: '48 kHz', io: '32 ch', status: 'Online' },
  { name: 'Director Control', ip: '192.168.1.100', type: 'Controller', sampleRate: 'N/A', io: 'Operator UI', status: 'Online' },
  { name: 'Engineer Tablet', ip: '192.168.1.101', type: 'Controller', sampleRate: 'N/A', io: 'Remote UI', status: 'Online' },
  { name: 'Backup Node', ip: '192.168.1.203', type: 'Service Node', sampleRate: '48 kHz', io: 'Snapshots', status: 'Online' }
];

export const logs: LogRecord[] = [
  { time: '10:24:15', level: 'INFO', source: 'System', message: 'Connected to WING RACK' },
  { time: '10:24:14', level: 'INFO', source: 'OSC', message: 'OSC connection established' },
  { time: '10:24:13', level: 'SUCCESS', source: 'Audio', message: 'Audio stream started at 48 kHz' },
  { time: '10:24:10', level: 'WARN', source: 'Stage Box', message: 'Redundant link not available' },
  { time: '10:24:09', level: 'INFO', source: 'Stage Box', message: 'Stage Box 2 connected' },
  { time: '10:24:05', level: 'SUCCESS', source: 'System', message: 'All systems ready' }
];

export const buses: BusRecord[] = [
  { name: 'Drums Bus', role: 'Group', level: -12.1, lufs: -17.4, channels: 8, color: 'green' },
  { name: 'Bass Bus', role: 'Group', level: -14.2, lufs: -18.8, channels: 2, color: 'yellow' },
  { name: 'Guitars Bus', role: 'Group', level: -15.8, lufs: -19.2, channels: 2, color: 'blue' },
  { name: 'Vocals Bus', role: 'Group', level: -10.8, lufs: -15.6, channels: 3, color: 'purple' },
  { name: 'FX Returns', role: 'Return', level: -24.4, lufs: -31.0, channels: 4, color: 'blue' }
];

export const trainingSessions = [
  { name: 'Live club vocal balance', status: 'Complete', examples: 128, score: '94%', updated: '10:22' },
  { name: 'Rock drums transient map', status: 'Running', examples: 84, score: '89%', updated: '10:19' },
  { name: 'Feedback prevention set', status: 'Queued', examples: 42, score: 'N/A', updated: '10:11' },
  { name: 'Low stage bleed model', status: 'Complete', examples: 212, score: '91%', updated: '09:58' },
  { name: 'Guitar masking correction', status: 'Review', examples: 76, score: '87%', updated: '09:44' },
  { name: 'Wing routing recognition', status: 'Complete', examples: 54, score: '96%', updated: '09:31' }
];

export const rules = [
  { name: 'Lead vocal sits above guitars', category: 'Balance', confidence: '98%', state: 'Active' },
  { name: 'Snare top HPF on small stage', category: 'EQ', confidence: '92%', state: 'Active' },
  { name: 'Bass DI resonance watch', category: 'Spectral', confidence: '87%', state: 'Review' },
  { name: 'No boost without headroom', category: 'Safety', confidence: '100%', state: 'Locked' },
  { name: 'Stereo pair linked fader moves', category: 'Control', confidence: '95%', state: 'Active' },
  { name: 'Supervised approval before write', category: 'Safety', confidence: '100%', state: 'Locked' },
  { name: 'Vocal FX send follows phrase density', category: 'FX', confidence: '90%', state: 'Active' },
  { name: 'Kick and bass split low-mid energy', category: 'Balance', confidence: '89%', state: 'Active' }
];

export const queueStages = [
  { name: 'Analyzer', count: 8, tone: 'blue', load: -12 },
  { name: 'Decision', count: 6, tone: 'purple', load: -18 },
  { name: 'Safety Gate', count: 4, tone: 'green', load: -24 },
  { name: 'Operator Review', count: 3, tone: 'yellow', load: -30 }
];

export const memoryTypes = [
  { name: 'Mix Rules', records: '4,820', freshness: '98%', level: -8 },
  { name: 'EQ Knowledge', records: '3,210', freshness: '92%', level: -14 },
  { name: 'Dynamics', records: '2,440', freshness: '89%', level: -19 },
  { name: 'Routing Profiles', records: '1,128', freshness: '99%', level: -11 },
  { name: 'Safety Decisions', records: '2,916', freshness: '100%', level: -6 }
];

export const settingsSections = [
  {
    name: 'Safety',
    rows: ['Supervised writes Enabled', 'Auto rollback Enabled', 'Max fader 0 dB', 'Feedback guard Armed']
  },
  {
    name: 'Audio Engine',
    rows: ['Sample rate 48 kHz', 'Buffer 128 samples', 'Metering 60 fps', 'Snapshot interval 30 sec']
  },
  {
    name: 'AI Director',
    rows: ['Mode Assist', 'Confidence gate 87%', 'Learning Local only', 'Broadcast Disabled']
  }
];

export const deviceHealth = [
  { name: 'Audio', value: 'Good', latency: '0.8 ms', tone: 'green' },
  { name: 'OSC', value: 'Good', latency: '1.2 ms', tone: 'blue' },
  { name: 'Network', value: 'Stable', latency: '0% loss', tone: 'green' },
  { name: 'Clock', value: 'Locked', latency: '48 kHz', tone: 'purple' }
];
