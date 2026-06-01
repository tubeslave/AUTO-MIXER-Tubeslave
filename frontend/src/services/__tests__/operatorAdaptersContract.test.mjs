import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import ts from 'typescript';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcRoot = path.resolve(__dirname, '../..');

async function importTypeScript(relativePath) {
  const sourcePath = path.join(srcRoot, relativePath);
  const source = await fs.readFile(sourcePath, 'utf8');
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ES2022,
      target: ts.ScriptTarget.ES2022,
    },
    fileName: sourcePath,
  });
  const encoded = Buffer.from(
    `${outputText}\n//# sourceURL=${pathToFileURL(sourcePath).href}`,
    'utf8',
  ).toString('base64');
  return import(`data:text/javascript;base64,${encoded}`);
}

test('adaptTasks preserves SafeGain ensemble evidence for read-only cockpit display', async () => {
  const { adaptTasks } = await importTypeScript('services/operatorAdapters.ts');

  const [task] = adaptTasks(
    {
      pending_actions: [
        {
          id: 'proposal-safe-gain-1',
          title: 'SafeGain trim review',
          target: 'Lead Vox',
          severity: 'high',
          status: 'pending',
          source: 'SafeGain',
          reason:
            'SafeGain measured input headroom. Ensemble context inactive (ensemble_time_context_stale); time stale; sources 1/2; lag 10/2 samples; operator review only.',
          requested_change: {
            value_type: 'gain',
            channel: 12,
            value: 3,
            safety_evidence: [
              {
                label: 'Ensemble context',
                value: 'ensemble_time_context_stale',
                reason: 'ensemble_time_context_stale',
              },
              {
                label: 'Ensemble time coherence',
                value: 'stale',
                reason: 'ensemble_time_context_stale',
              },
              {
                label: 'Ensemble time sources',
                value: 1,
                target: 2,
              },
              {
                label: 'Ensemble lag samples',
                value: 10,
                target: 2,
              },
              {
                label: 'Ensemble apply policy',
                value: 'operator_review_only',
              },
              {
                label: 'Distortion risk',
                value: 'low',
              },
            ],
          },
          can_apply: true,
        },
      ],
    },
    [],
  );

  assert.equal(task.reason.includes('Ensemble context inactive'), true);
  assert.deepEqual(
    task.safetyEvidence.map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
      'Distortion risk',
    ],
  );
  assert.equal(task.safetyEvidence[0].value, 'ensemble time context stale');
  assert.equal(task.safetyEvidence[0].tone, 'yellow');
  assert.equal(task.safetyEvidence[2].value, '1/2');
  assert.equal(task.safetyEvidence[3].value, '10/2');
  assert.equal(task.safetyEvidence[4].tone, 'yellow');
  assert.equal(task.safetyEvidence[5].tone, 'green');
});

test('safetyEvidenceForTask keeps ensemble rows visible before generic evidence', async () => {
  const { ensembleEvidenceForTask, safetyEvidenceForTask } = await importTypeScript('mock/automixerData.ts');

  const task = {
    id: 'proposal-safe-gain-1',
    priority: 'High',
    title: 'SafeGain trim review',
    agent: 'SafeGain',
    status: 'Pending',
    time: '10:31',
    safetyEvidence: [
      { label: 'Distortion risk', value: 'low', tone: 'green' },
      { label: 'Ensemble context', value: 'ensemble time context stale', tone: 'yellow' },
      { label: 'Preamp zone', value: 'clean', tone: 'green' },
      { label: 'Ensemble time coherence', value: 'stale', tone: 'yellow' },
      { label: 'Ensemble time sources', value: '1/2', tone: 'yellow' },
      { label: 'Readback', value: 'required', tone: 'yellow' },
      { label: 'Ensemble lag samples', value: '10/2', tone: 'yellow' },
      { label: 'Rollback', value: 'required', tone: 'yellow' },
      { label: 'Ensemble apply policy', value: 'operator review only', tone: 'yellow' },
    ],
  };

  const evidence = safetyEvidenceForTask(task, 'Supervised');

  assert.deepEqual(
    evidence.map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
      'Distortion risk',
    ],
  );

  assert.deepEqual(
    ensembleEvidenceForTask(task).map(row => row.label),
    [
      'Ensemble context',
      'Ensemble time coherence',
      'Ensemble time sources',
      'Ensemble lag samples',
      'Ensemble apply policy',
    ],
  );
});

test('adaptChannels treats control_state mute as read-only channel status', async () => {
  const { adaptChannels } = await importTypeScript('services/operatorAdapters.ts');

  const fallback = [{
    id: 12,
    name: 'Lead Vocal',
    type: 'Vocal Lead',
    source: 'In 12',
    level: -18,
    lufs: -22,
    dynamic: 6,
    status: 'ACTIVE',
    recognition: 'Lead Vocal',
    group: 'VOCALS',
    fader: -10,
    confidence: 96,
  }];

  const [muted] = adaptChannels(
    {
      channels: [{
        id: 12,
        name: 'Lead Vocal',
        role: 'Vocal Lead',
        group: 'Vocals',
        status: 'active',
        level_db: -18,
        lufs: -22,
        dynamic_range_db: 6,
        control_state: { fader_db: -10, mute: 1 },
        ai_recognition: { label: 'Lead Vocal', confidence: 96 },
      }],
    },
    fallback,
  );

  assert.equal(muted.status, 'MUTED');
  assert.equal(muted.fader, -10);

  const [unmuted] = adaptChannels(
    {
      channels: [{
        id: 12,
        name: 'Lead Vocal',
        role: 'Vocal Lead',
        group: 'Vocals',
        status: 'active',
        level_db: -18,
        lufs: -22,
        dynamic_range_db: 6,
        control_state: { fader_db: -10, mute: '0' },
        ai_recognition: { label: 'Lead Vocal', confidence: 96 },
      }],
    },
    fallback,
  );

  assert.equal(unmuted.status, 'ACTIVE');
});

test('directorDecisionRows preserves backend task status instead of forcing applied', async () => {
  const { directorDecisionRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const rows = directorDecisionRows(
    [
      {
        id: 'proposal-safe-gain-1',
        priority: 'High',
        title: 'SafeGain trim review',
        agent: 'SafeGain',
        status: 'Pending',
        time: '10:31',
        target: 'CH 12',
      },
      {
        id: 'proposal-eq-1',
        priority: 'Medium',
        title: 'EQ proposal remains review-only',
        agent: 'AutoEQ',
        status: 'Review',
        time: '10:32',
      },
      {
        id: 'proposal-applied-1',
        priority: 'Low',
        title: 'Operator-approved fader down',
        agent: 'AutoFader',
        status: 'Applied',
        time: '10:33',
      },
    ],
    [
      {
        time: '10:00',
        title: 'Fallback decision',
        target: 'CH 01',
        status: 'Applied',
        tone: 'green',
      },
    ],
  );

  assert.deepEqual(
    rows.map(row => row.status),
    ['Pending', 'Review', 'Applied'],
  );
  assert.equal(rows[0].tone, 'yellow');
  assert.equal(rows[1].tone, 'yellow');
  assert.equal(rows[2].tone, 'green');
  assert.equal(rows[1].target, 'proposal-eq-1');
});

test('directorDecisionRows keeps mock fallback only when no backend tasks exist', async () => {
  const { directorDecisionRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallback = [
    {
      time: '10:00',
      title: 'Fallback decision',
      target: 'CH 01',
      status: 'Review',
      tone: 'purple',
    },
  ];

  assert.deepEqual(directorDecisionRows([], fallback), fallback);
});

test('toneForDirectorStatus keeps non-applied director states visually distinct', async () => {
  const { toneForDirectorStatus } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  assert.equal(toneForDirectorStatus('Pending', 'Low'), 'yellow');
  assert.equal(toneForDirectorStatus('Queued'), 'yellow');
  assert.equal(toneForDirectorStatus('Blocked'), 'red');
  assert.equal(toneForDirectorStatus('Failed'), 'red');
  assert.equal(toneForDirectorStatus('Safe'), 'green');
  assert.equal(toneForDirectorStatus('Approved'), 'green');
});

test('directorProposalRows derives Agent Suggestions from backend task rows', async () => {
  const { directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const rows = directorProposalRows(
    [
      {
        id: 'proposal-safe-gain-1',
        priority: 'High',
        title: 'SafeGain trim review',
        agent: 'SafeGain',
        status: 'Pending',
        time: '10:31',
        target: 'CH 12',
        reason: 'Operator review required before any supervised gate apply.',
        requestedValueType: 'gain',
      },
      {
        id: 'proposal-eq-1',
        priority: 'Medium',
        title: 'EQ remains proposal-only',
        agent: 'AutoEQ',
        status: 'Review',
        time: '10:32',
        reviewOnly: true,
        canApply: false,
        backendStatus: 'proposal_only',
      },
    ],
    [
      {
        id: 'fallback',
        agent: 'Mock',
        title: 'Fallback suggestion',
        detail: 'UI fallback',
        status: 'Review',
        tone: 'purple',
        reviewMessage: 'fallback reviewed',
        rejectMessage: 'fallback rejected',
      },
    ],
  );

  assert.deepEqual(
    rows.map(row => row.id),
    ['proposal-safe-gain-1', 'proposal-eq-1'],
  );
  assert.equal(rows[0].agent, 'SafeGain');
  assert.equal(rows[0].status, 'Pending');
  assert.equal(rows[0].tone, 'yellow');
  assert.equal(rows[0].detail, 'Operator review required before any supervised gate apply.');
  assert.equal(rows[0].reviewMessage.includes('No OSC write sent'), true);
  assert.equal(rows[1].detail, 'proposal_only - review only');
  assert.equal(rows[1].status, 'Review only');
  assert.equal(rows[1].tone, 'red');
});

test('directorProposalRows keeps hardcoded suggestions only without backend tasks', async () => {
  const { directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallback = [
    {
      id: 'fallback',
      agent: 'Mock',
      title: 'Fallback suggestion',
      detail: 'UI fallback',
      status: 'Review',
      tone: 'purple',
      reviewMessage: 'fallback reviewed',
      rejectMessage: 'fallback rejected',
    },
  ];

  assert.deepEqual(directorProposalRows([], fallback), fallback);
});

test('directorProposalRows suppresses fallback suggestions for backend source', async () => {
  const { directorFallbackRowsForSource, directorDecisionRows, directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallback = [
    {
      id: 'fallback',
      agent: 'Mock',
      title: 'Fallback suggestion',
      detail: 'UI fallback',
      status: 'Review',
      tone: 'purple',
      reviewMessage: 'fallback reviewed',
      rejectMessage: 'fallback rejected',
    },
  ];
  const fallbackDecision = [
    {
      time: '10:00',
      title: 'Fallback decision',
      target: 'CH 01',
      status: 'Review',
      tone: 'purple',
    },
  ];

  assert.deepEqual(directorFallbackRowsForSource('Backend', fallback), []);
  assert.deepEqual(directorFallbackRowsForSource('Mock', fallback), fallback);
  assert.deepEqual(directorProposalRows([], fallback, 'Backend'), []);
  assert.deepEqual(directorDecisionRows([], fallbackDecision, 'Backend'), []);
  assert.deepEqual(directorProposalRows([], fallback, 'Mock'), fallback);
});

test('directorEventRows derives backend events from runtime state instead of mock rows', async () => {
  const { directorEventRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallbackEvents = [
    {
      time: '10:31:08',
      source: 'Telemetry',
      message: 'Mock meter frame accepted for dashboard render.',
      tone: 'blue',
    },
  ];
  const runtime = {
    currentGoal: 'Keep writes supervised.',
    readyLabel: 'Not live ready',
    readyTone: 'red',
    audioLabel: 'Read-only',
    audioTone: 'yellow',
    proposalQueueLabel: '2 pending',
    proposalQueueTone: 'yellow',
    liveWritePolicy: 'supervised_fader_gain_only',
    manualGateLabel: 'Manual gate armed',
    manualGateTone: 'green',
    resetRequired: false,
    hardwareProofItems: [
      { label: 'Readback', value: 'cache only', tone: 'yellow' },
      { label: 'Rollback', value: 'required', tone: 'yellow' },
    ],
    blockedReasons: ['physical_pre_read_required'],
    statusItems: [],
    sourceLabel: 'Backend',
  };

  const rows = directorEventRows(runtime, fallbackEvents);

  assert.deepEqual(
    rows.map(row => row.source),
    ['Safety', 'Readiness', 'Blockers'],
  );
  assert.equal(rows.some(row => row.message.includes('Mock meter frame')), false);
  assert.equal(rows[0].message.includes('supervised_fader_gain_only'), true);
  assert.equal(rows[2].message, 'physical_pre_read_required');
});

test('directorEventRows keeps static events only for mock source', async () => {
  const { directorEventRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const fallbackEvents = [
    {
      time: '10:31:08',
      source: 'Telemetry',
      message: 'Mock meter frame accepted for dashboard render.',
      tone: 'blue',
    },
  ];
  const runtime = {
    currentGoal: 'Mock goal',
    readyLabel: 'Mock only',
    readyTone: 'red',
    audioLabel: 'Mock meters',
    audioTone: 'yellow',
    proposalQueueLabel: 'Mock queue',
    proposalQueueTone: 'blue',
    liveWritePolicy: 'mock_frontend_only',
    manualGateLabel: 'UI only',
    manualGateTone: 'yellow',
    resetRequired: false,
    hardwareProofItems: [],
    blockedReasons: [],
    statusItems: [],
    sourceLabel: 'Mock',
  };

  assert.deepEqual(directorEventRows(runtime, fallbackEvents), fallbackEvents);
});

test('directorProposalRows excludes completed backend tasks from active suggestions', async () => {
  const { directorProposalRows } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const rows = directorProposalRows(
    [
      {
        id: 'proposal-applied-1',
        priority: 'Low',
        title: 'Operator-approved fader down',
        agent: 'AutoFader',
        status: 'Applied',
        time: '10:33',
      },
      {
        id: 'proposal-rejected-1',
        priority: 'Medium',
        title: 'Rejected EQ change',
        agent: 'AutoEQ',
        status: 'Rejected',
        time: '10:34',
      },
      {
        id: 'proposal-review-1',
        priority: 'High',
        title: 'Review-only SafeGain evidence',
        agent: 'SafeGain',
        status: 'Review',
        time: '10:35',
        reviewOnly: true,
        canApply: false,
        backendStatus: 'proposal_only',
      },
    ],
    [
      {
        id: 'fallback',
        agent: 'Mock',
        title: 'Fallback suggestion',
        detail: 'UI fallback',
        status: 'Review',
        tone: 'purple',
        reviewMessage: 'fallback reviewed',
        rejectMessage: 'fallback rejected',
      },
    ],
  );

  assert.deepEqual(
    rows.map(row => row.id),
    ['proposal-review-1'],
  );
  assert.equal(rows[0].detail, 'proposal_only - review only');
  assert.equal(rows[0].status, 'Review only');
  assert.equal(rows[0].tone, 'red');
});

test('directorProposalRows treats apply blockers as review-only safety evidence', async () => {
  const { directorProposalRows, taskRequiresReviewOnly } = await importTypeScript('components/dashboard/directorPanelModel.ts');

  const blockedTask = {
    id: 'proposal-gain-proof-1',
    priority: 'High',
    title: 'Gain proof requires physical pre-read',
    agent: 'Product Layer',
    status: 'Review',
    time: '10:36',
    target: 'CH 09',
    canApply: true,
    backendStatus: 'blocked_safety_check',
    applyBlockers: ['physical_pre_read_required', 'rollback_readback_missing'],
    requestedValueType: 'gain',
  };

  const rows = directorProposalRows(
    [blockedTask],
    [
      {
        id: 'fallback',
        agent: 'Mock',
        title: 'Fallback suggestion',
        detail: 'UI fallback',
        status: 'Review',
        tone: 'purple',
        reviewMessage: 'fallback reviewed',
        rejectMessage: 'fallback rejected',
      },
    ],
  );

  assert.equal(taskRequiresReviewOnly(blockedTask), true);
  assert.equal(rows[0].status, 'Review only');
  assert.equal(rows[0].tone, 'red');
  assert.equal(rows[0].detail, 'physical_pre_read_required, rollback_readback_missing - review only');
  assert.equal(rows[0].reviewMessage.includes('No OSC write sent'), true);
});

test('dashboard director panel renders agent suggestions as review-only controls', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/dashboard/DirectorPanel.tsx'),
    'utf8',
  );

  assert.match(source, /directorProposalRows/);
  assert.match(source, /Просмотр/);
  assert.match(source, /No active backend proposals require review/);
  assert.match(source, /No OSC write sent/);
  assert.equal(source.includes('Принять'), false);
});

test('dashboard mixer header uses the selected channel dynamically', async () => {
  const source = await fs.readFile(
    path.join(srcRoot, 'components/dashboard/AutomixerDashboard.tsx'),
    'utf8',
  );

  assert.match(source, /selectedChannel/);
  assert.match(source, /Selected: \{selectedChannel \? `CH/);
  assert.match(source, /suppressMockSolo/);
  assert.equal(source.includes('Selected: Lead Vocal'), false);
});

test('dashboardTelemetryForChannel derives read-only strip state from channel snapshots', async () => {
  const { dashboardTelemetryForChannel } = await importTypeScript('components/dashboard/dashboardTelemetryModel.ts');

  const fallback = {
    waveform: [20, 40, 60],
    meterL: -18,
    meterR: -19,
    gateLabel: 'G -40',
    compLabel: 'C 2.0',
    safeState: 'FOCUS',
    pan: 0,
    muted: false,
    solo: true,
    accent: 'purple',
  };

  const muted = dashboardTelemetryForChannel(
    {
      id: 14,
      name: 'Backing Vocal',
      type: 'Vocal Back',
      source: 'In 14',
      level: -48,
      lufs: -55,
      dynamic: 2,
      status: 'MUTED',
      recognition: 'Backing Vocal',
      group: 'VOCALS',
      fader: -50,
      confidence: 95,
    },
    fallback,
  );

  assert.equal(muted.meterL, -48);
  assert.equal(muted.meterR, -48);
  assert.equal(muted.muted, true);
  assert.equal(muted.safeState, 'MUTE');
  assert.equal(muted.accent, 'red');
  assert.deepEqual(muted.waveform, fallback.waveform);
  assert.equal(muted.solo, true);

  const backendMuted = dashboardTelemetryForChannel(
    {
      id: 14,
      name: 'Backing Vocal',
      type: 'Vocal Back',
      source: 'In 14',
      level: -48,
      lufs: -55,
      dynamic: 2,
      status: 'MUTED',
      recognition: 'Backing Vocal',
      group: 'VOCALS',
      fader: -50,
      confidence: 95,
    },
    fallback,
    { suppressMockSolo: true },
  );

  assert.equal(backendMuted.solo, false);

  const active = dashboardTelemetryForChannel(
    {
      id: 12,
      name: 'Lead Vocal',
      type: 'Vocal Lead',
      source: 'In 12',
      level: -18,
      lufs: -18,
      dynamic: 6,
      status: 'ACTIVE',
      recognition: 'Lead Vocal',
      group: 'VOCALS',
      fader: -10,
      confidence: 98,
    },
    fallback,
  );

  assert.equal(active.safeState, 'OK');
  assert.equal(active.muted, false);
  assert.equal(active.accent, 'purple');

  const hot = dashboardTelemetryForChannel(
    {
      id: 12,
      name: 'Lead Vocal',
      type: 'Vocal Lead',
      source: 'In 12',
      level: -8.5,
      lufs: -18,
      dynamic: 6,
      status: 'ACTIVE',
      recognition: 'Lead Vocal',
      group: 'VOCALS',
      fader: -10,
      confidence: 98,
    },
    fallback,
  );

  assert.equal(hot.safeState, 'HOT');
  assert.equal(hot.accent, 'yellow');

  const clipping = dashboardTelemetryForChannel(
    {
      id: 13,
      name: 'Guest Mic',
      type: 'Vocal Lead',
      source: 'In 13',
      level: -2.5,
      lufs: -10,
      dynamic: 4,
      status: 'ACTIVE',
      recognition: 'Vocal',
      group: 'VOCALS',
      fader: -6,
      confidence: 90,
    },
    fallback,
  );

  assert.equal(clipping.safeState, 'CLIP');
  assert.equal(clipping.accent, 'red');
});

test('read-only DTO adapters preserve explicit empty backend snapshots', async () => {
  const {
    adaptBusesSnapshot,
    adaptCommunicationsSnapshot,
    adaptPerformanceRecords,
    adaptMemorySnapshot,
    adaptTrainingSnapshot,
    adaptRulesSnapshot,
    adaptDevices,
  } = await importTypeScript('services/operatorAdapters.ts');

  const fallbackBus = [{
    name: 'Mock Bus',
    role: 'Group',
    level: -12,
    lufs: -14,
    channels: 2,
    color: 'green',
  }];
  const fallbackCommunication = [{
    time: '10:00',
    from: 'Mock',
    to: 'Director',
    topic: 'Fallback message',
    status: 'Info',
    tone: 'blue',
  }];
  const fallbackMemory = [{
    name: 'Mock memory',
    type: 'Mock',
    size: '1 KB',
    confidence: 80,
    updated: 'mock',
  }];
  const fallbackTraining = [{
    name: 'Mock training',
    status: 'Queued',
    examples: 1,
    score: '80%',
    updated: 'mock',
  }];
  const fallbackRule = [{
    name: 'Mock rule',
    category: 'Mock',
    confidence: '80%',
    state: 'Review',
  }];
  const fallbackDevice = [{
    name: 'Mock Device',
    ip: '127.0.0.1',
    type: 'Console',
    sampleRate: '48 kHz',
    io: 'mock',
    status: 'Online',
  }];

  assert.deepEqual(adaptBusesSnapshot({ buses: [] }, fallbackBus), []);
  assert.deepEqual(adaptCommunicationsSnapshot({ communications: [] }, fallbackCommunication), []);
  assert.deepEqual(adaptPerformanceRecords({ agents: [] }, {}, [{
    agent: 'Mock Agent',
    cpu: '1%',
    memory: '1 MB',
    decisions: 1,
    success: '80%',
    latency: '1 ms',
    status: 'Good',
  }]), []);
  assert.deepEqual(adaptMemorySnapshot({ memories: [] }, fallbackMemory), []);
  assert.deepEqual(adaptTrainingSnapshot({ sessions: [] }, fallbackTraining), []);
  assert.deepEqual(adaptRulesSnapshot({ rules: [] }, fallbackRule), []);
  assert.deepEqual(adaptDevices({ connected_devices: [], nodes: [] }, fallbackDevice), []);

  assert.deepEqual(adaptBusesSnapshot(undefined, fallbackBus), fallbackBus);
  assert.deepEqual(adaptCommunicationsSnapshot(undefined, fallbackCommunication), fallbackCommunication);
  assert.deepEqual(adaptPerformanceRecords(undefined, {}, [{
    agent: 'Mock Agent',
    cpu: '1%',
    memory: '1 MB',
    decisions: 1,
    success: '80%',
    latency: '1 ms',
    status: 'Good',
  }]).length, 1);
  assert.deepEqual(adaptMemorySnapshot(undefined, fallbackMemory), fallbackMemory);
  assert.deepEqual(adaptTrainingSnapshot(undefined, fallbackTraining), fallbackTraining);
  assert.deepEqual(adaptRulesSnapshot(undefined, fallbackRule), fallbackRule);
  assert.deepEqual(adaptDevices(undefined, fallbackDevice), fallbackDevice);
});

test('cockpit marks visual placeholder metrics instead of presenting them as live telemetry', async () => {
  const dashboardSource = await fs.readFile(path.join(srcRoot, 'components/dashboard/AutomixerDashboard.tsx'), 'utf8');
  const performanceSource = await fs.readFile(path.join(srcRoot, 'views/PerformanceView.tsx'), 'utf8');
  const systemSource = await fs.readFile(path.join(srcRoot, 'views/SystemViews.tsx'), 'utf8');
  const backendHookSource = await fs.readFile(path.join(srcRoot, 'services/useOperatorBackend.ts'), 'utf8');
  const shellSource = await fs.readFile(path.join(srcRoot, 'components/prototype/AutomixerComponents.tsx'), 'utf8');
  const agentsSource = await fs.readFile(path.join(srcRoot, 'components/prototype/AgentsConsolePanels.tsx'), 'utf8');

  assert.match(dashboardSource, /Backend DTO/);
  assert.match(dashboardSource, /Display-only/);
  assert.equal(dashboardSource.includes('BS.1770 mock readout'), false);
  assert.match(backendHookSource, /performance: payloadSource\(payloads\.agents/);
  assert.match(performanceSource, /data\.readModelSources\.performance/);
  assert.match(performanceSource, /rowsForReadModelSource/);
  assert.equal(performanceSource.includes("data.performanceRecords.length ? 'Backend-derived' : 'Mock'"), false);
  assert.match(performanceSource, /Display Trace/);
  assert.match(performanceSource, /Trend traces/);
  assert.match(systemSource, /System Latency · Display-only/);
  assert.match(systemSource, /Activity trace is display-only/);
  assert.match(shellSource, /Display-only health/);
  assert.match(agentsSource, /Traffic by Agent · Display Trace/);
});
