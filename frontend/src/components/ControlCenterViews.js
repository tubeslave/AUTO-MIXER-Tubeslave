import React from 'react';
import './ControlCenterViews.css';

const mixChannels = [
  { id: 1, name: 'Kick In', role: 'Kick', group: 'DRUMS', level: -12.4, lufs: -14.2, dynamic: 8.1, confidence: 98 },
  { id: 2, name: 'Kick Out', role: 'Kick', group: 'DRUMS', level: -14.1, lufs: -16.3, dynamic: 7.8, confidence: 97 },
  { id: 3, name: 'Snare Top', role: 'Snare', group: 'DRUMS', level: -11.8, lufs: -13.1, dynamic: 9.3, confidence: 99 },
  { id: 4, name: 'Snare Bottom', role: 'Snare', group: 'DRUMS', level: -13.6, lufs: -15.4, dynamic: 8.6, confidence: 96 },
  { id: 5, name: 'Hi-Hat', role: 'Hi-Hat', group: 'DRUMS', level: -16.3, lufs: -18.9, dynamic: 6.2, confidence: 95 },
  { id: 6, name: 'Tom 1', role: 'Tom', group: 'DRUMS', level: -13.0, lufs: -15.7, dynamic: 8.4, confidence: 94 },
  { id: 7, name: 'Tom 2', role: 'Tom', group: 'DRUMS', level: -14.2, lufs: -16.0, dynamic: 8.0, confidence: 94 },
  { id: 8, name: 'OH L', role: 'Overhead', group: 'DRUMS', level: -18.1, lufs: -20.3, dynamic: 10.2, confidence: 98 },
  { id: 9, name: 'OH R', role: 'Overhead', group: 'DRUMS', level: -18.0, lufs: -20.1, dynamic: 10.5, confidence: 98 },
  { id: 10, name: 'Bass DI', role: 'Bass', group: 'BASS', level: -10.5, lufs: -11.8, dynamic: 7.1, confidence: 99 },
  { id: 11, name: 'Bass Amp', role: 'Bass', group: 'BASS', level: -14.9, lufs: -16.8, dynamic: 6.9, confidence: 97 },
  { id: 12, name: 'Lead Vocal', role: 'Vocal Lead', group: 'VOCALS', level: -9.3, lufs: -10.2, dynamic: 6.3, confidence: 99 },
  { id: 13, name: 'Backing Vocal 1', role: 'Vocal Back', group: 'VOCALS', level: -16.5, lufs: -17.9, dynamic: 5.4, confidence: 96 },
  { id: 14, name: 'Backing Vocal 2', role: 'Vocal Back', group: 'VOCALS', level: -16.7, lufs: -18.0, dynamic: 5.6, confidence: 95 },
  { id: 15, name: 'Gtr 1', role: 'Guitar Elec', group: 'GUITARS', level: -12.7, lufs: -14.6, dynamic: 7.3, confidence: 97 },
  { id: 16, name: 'Gtr 2', role: 'Guitar Elec', group: 'GUITARS', level: -13.1, lufs: -15.1, dynamic: 7.6, confidence: 96 },
];

const agents = [
  { name: 'Analyzer Agent', role: 'Анализирует каналы и спектр', status: 'ACTIVE', task: 'Анализ 14 каналов', confidence: 92, tone: 'green' },
  { name: 'Decision Agent', role: 'Принимает решения по обработке', status: 'ACTIVE', task: 'Оценивает баланс микса', confidence: 87, tone: 'green' },
  { name: 'Mix Critic Agent', role: 'Критически оценивает микс', status: 'ACTIVE', task: 'Проверяет Lead Vocal', confidence: 89, tone: 'green' },
  { name: 'Spectral Agent', role: 'Следит за спектральным балансом', status: 'ACTIVE', task: 'Мониторинг спектра', confidence: 81, tone: 'green' },
  { name: 'Safety Agent', role: 'Контролирует безопасность', status: 'ACTIVE', task: 'Проверка предложений', confidence: 95, tone: 'green' },
  { name: 'OSC Controller Agent', role: 'Готовит команды микшера', status: 'ACTIVE', task: 'Ожидает подтверждения', confidence: 100, tone: 'green' },
  { name: 'Learning Agent', role: 'Обучается на решениях', status: 'ACTIVE', task: 'Формирует правило', confidence: 76, tone: 'yellow' },
  { name: 'Memory Agent', role: 'Синхронизирует память', status: 'ACTIVE', task: 'База знаний обновлена', confidence: 90, tone: 'green' },
];

const decisions = [
  ['10:24', 'Lead Vocal: -1.5 dB fader', 'APPLIED'],
  ['10:23', 'Snare Top: HPF 120 Hz', 'APPLIED'],
  ['10:22', 'Bass DI: +2 dB 80 Hz', 'APPLIED'],
  ['10:21', 'Overheads: Align phase', 'APPLIED'],
  ['10:20', 'Gtr 1: -2 dB at 3.2 kHz', 'APPLIED'],
];

function formatMetric(value, fallback = 'N/A', suffix = '') {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return fallback;
  return `${Number(value).toFixed(1)}${suffix}`;
}

function normalizeChannel(channel) {
  const level = channel.level_db ?? channel.level ?? channel.meter?.rms_db ?? -60;
  const peak = channel.peak_db ?? channel.meter?.peak_db ?? null;
  const dynamic = channel.dynamic_range_db ?? channel.dynamic ?? (
    peak !== null && level !== null ? Math.max(0, Number(peak) - Number(level)) : 0
  );
  return {
    id: channel.id ?? channel.number,
    name: channel.name || `Ch ${channel.number || channel.id}`,
    role: channel.role || channel.routing_role || 'Channel',
    group: channel.group || 'CHANNELS',
    level: Number.isFinite(Number(level)) ? Number(level) : -60,
    peak,
    lufs: channel.lufs ?? channel.meter?.lufs ?? null,
    dynamic: Number.isFinite(Number(dynamic)) ? Number(dynamic) : 0,
    confidence: channel.ai_recognition?.confidence ?? channel.confidence ?? 0,
    status: channel.status || 'unknown',
    source: channel.source || `In ${String(channel.id ?? channel.number).padStart(2, '0')}`,
  };
}

function channelsFromInventory(inventory) {
  if (inventory?.channels?.length) {
    return inventory.channels.map(normalizeChannel);
  }
  return mixChannels;
}

function decisionsFromQueue(queue) {
  const pending = queue?.pending_actions || [];
  const history = queue?.history || [];
  const rows = [...pending, ...history].slice(0, 5).map((item, index) => [
    item.created_at ? String(item.created_at).slice(11, 16) : `--:${String(index).padStart(2, '0')}`,
    item.title || item.message || item.action || 'Decision',
    String(item.status || 'PENDING').toUpperCase(),
  ]);
  return rows.length ? rows : decisions;
}

function statusClass(status) {
  if (['ACTIVE', 'OK', 'ONLINE', 'RUNNING'].includes(String(status).toUpperCase())) return 'state-green';
  if (['OFFLINE', 'UNAVAILABLE', 'BLOCKED'].includes(String(status).toUpperCase())) return 'state-red';
  return 'state-yellow';
}

function MiniSparkline({ tone = 'green' }) {
  return (
    <div className={`mini-spark mini-spark-${tone}`}>
      <i style={{ height: '30%' }} />
      <i style={{ height: '65%' }} />
      <i style={{ height: '45%' }} />
      <i style={{ height: '80%' }} />
      <i style={{ height: '38%' }} />
      <i style={{ height: '72%' }} />
      <i style={{ height: '50%' }} />
      <i style={{ height: '88%' }} />
      <i style={{ height: '42%' }} />
      <i style={{ height: '62%' }} />
    </div>
  );
}

function MeterBar({ value }) {
  const pct = Math.max(8, Math.min(100, ((value + 60) / 60) * 100));
  return (
    <div className="meter-track">
      <span style={{ width: `${pct}%` }} />
    </div>
  );
}

function DonutScore({ value, label }) {
  return (
    <div className="donut-score" style={{ '--score': `${value * 3.6}deg` }}>
      <div>
        <strong>{value}%</strong>
        <span>{label}</span>
      </div>
    </div>
  );
}

function SpectrumChart({ compact = false }) {
  return (
    <div className={`spectrum-chart ${compact ? 'compact' : ''}`}>
      <svg viewBox="0 0 220 82" role="img" aria-label="Spectrum">
        <defs>
          <linearGradient id="spectrumFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#a371f7" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#1f6feb" stopOpacity="0.04" />
          </linearGradient>
          <linearGradient id="spectrumLine" x1="0" x2="1">
            <stop offset="0%" stopColor="#58a6ff" />
            <stop offset="55%" stopColor="#a371f7" />
            <stop offset="100%" stopColor="#58a6ff" />
          </linearGradient>
        </defs>
        <path d="M0 70 L0 50 L10 55 L20 32 L30 49 L40 25 L50 43 L60 28 L70 39 L80 24 L90 34 L100 20 L110 30 L120 26 L130 18 L140 34 L150 24 L160 38 L170 30 L180 45 L190 36 L200 50 L210 44 L220 62 L220 70 Z" fill="url(#spectrumFill)" />
        <polyline points="0,50 10,55 20,32 30,49 40,25 50,43 60,28 70,39 80,24 90,34 100,20 110,30 120,26 130,18 140,34 150,24 160,38 170,30 180,45 190,36 200,50 210,44 220,62" fill="none" stroke="url(#spectrumLine)" strokeWidth="2" />
        <polyline points="0,58 14,42 27,53 42,36 56,50 70,31 84,45 98,33 112,40 126,27 140,38 154,32 168,46 182,39 196,54 210,48 220,66" fill="none" stroke="#f778ba" strokeOpacity="0.72" strokeWidth="1.3" />
      </svg>
    </div>
  );
}

function DirectorPanel({ decisionQueue, operatorModeStatus, proposalActions }) {
  const rows = decisionsFromQueue(decisionQueue);
  const queueAvailable = decisionQueue?.proposal_queue_available !== false;
  const agentRuntimeAvailable = decisionQueue?.agent_runtime_available === true;
  const canApply = operatorModeStatus?.capabilities?.can_apply_to_console === true;
  const modeLabel = operatorModeStatus?.label || 'Assist';
  const currentGoal = queueAvailable
    ? 'Обеспечить читаемость вокала и сбалансированный микс в рамках текущей политики режима.'
    : 'Очередь предложений недоступна; экран показывает read-only состояние backend без фиктивных применений.';
  return (
    <aside className="director-panel">
      <section className="director-card director-profile">
        <div className="director-avatar">AI</div>
        <div>
          <h3>Директор Automixer</h3>
          <p>Mode: {modeLabel}</p>
          <span className={queueAvailable ? 'online-dot' : 'state-yellow'}>
            {agentRuntimeAvailable ? 'Agent online' : queueAvailable ? 'Queue online' : 'Queue unavailable'}
          </span>
        </div>
      </section>
      <section className="director-card">
        <h4>Текущая цель</h4>
        <p>{currentGoal}</p>
        <button className="ghost-action">Детали цели</button>
      </section>
      <section className="director-card">
        <h4>Последние решения</h4>
        {rows.map(([time, text, state]) => (
          <div className="decision-row" key={`${time}-${text}`}>
            <span>{time}</span>
            <p>{text}</p>
            <strong>{state}</strong>
          </div>
        ))}
      </section>
      <section className="director-card">
        <h4>Предложения агента</h4>
        {(decisionQueue?.pending_actions || []).slice(0, 2).map((item) => (
          <div className="proposal-row" key={item.id || item.title}>
            <p>{item.title}</p>
            <div>
              <button onClick={() => proposalActions?.onAccept?.(item.id)}>Принять</button>
              {canApply && item.can_apply && (
                <button onClick={() => proposalActions?.onApply?.(item.id)}>Применить</button>
              )}
              <button onClick={() => proposalActions?.onDismiss?.(item.id)}>Отклонить</button>
            </div>
          </div>
        ))}
        {(!decisionQueue?.pending_actions || decisionQueue.pending_actions.length === 0) && (
          <div className="proposal-row">
            <p>{decisionQueue?.reason || 'Нет предложений в очереди'}</p>
          </div>
        )}
      </section>
    </aside>
  );
}

function ChannelStrip({ channel, selected }) {
  const meterHeight = Math.max(8, Math.min(92, 80 + channel.level));
  const faderPosition = Math.max(8, Math.min(84, 62 + channel.level));
  return (
    <div className={`console-strip ${selected ? 'selected' : ''}`}>
      <div className="strip-head">
        <strong>{channel.name}</strong>
        <span>{channel.role}</span>
      </div>
      <b className="strip-db">{formatMetric(channel.level, 'N/A', ' dB')}</b>
      <MiniSparkline tone={channel.group === 'VOCALS' ? 'purple' : 'green'} />
      <div className="strip-tools"><span>GATE</span><span>COMP</span></div>
      <div className="strip-fader">
        <div className="strip-meter"><span style={{ height: `${meterHeight}%` }} /></div>
        <div className="fader-rail"><i style={{ bottom: `${faderPosition}%` }} /></div>
      </div>
      <div className="strip-buttons"><button>M</button><button>S</button></div>
      <footer>{String(channel.id).padStart(2, '0')}</footer>
    </div>
  );
}

function MetricCard({ title, children }) {
  return (
    <section className="metric-card">
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export function DashboardView({ snapshot, channelInventory, decisionQueue, operatorModeStatus, proposalActions, operatorAnalysisReport }) {
  const channels = channelsFromInventory(channelInventory);
  const master = snapshot?.master_bus || {};
  const channelSummary = snapshot?.channel_summary || channelInventory?.summary || {};
  const decisionSummary = snapshot?.decision_summary || decisionQueue?.summary || {};
  const aiConfidence = decisionQueue?.agent_runtime_available === true ? 87 : 0;
  return (
    <div className="console-page console-page-dashboard">
      <div className="console-main">
        <div className="metrics-grid">
          <MetricCard title="Master Bus">
            <div className="master-values">
              <div className="big-metric">{formatMetric(master.integrated_lufs, 'N/A')} <span>LUFS</span></div>
              <div className="big-metric small">{formatMetric(master.true_peak_db, 'N/A')} <span>dBTP</span></div>
            </div>
            <div className="master-labels"><span>Integrated</span><span>True Peak</span></div>
            <MeterBar value={master.integrated_lufs ?? -60} />
          </MetricCard>
          <MetricCard title="Spectrum">
            <SpectrumChart />
            <div className="spectrum-axis"><span>20</span><span>100</span><span>1k</span><span>10k</span></div>
          </MetricCard>
          <MetricCard title="Loudness">
            <div className="metric-list"><span>Short Term</span><strong>{formatMetric(master.short_term_lufs, 'N/A', ' LUFS')}</strong></div>
            <div className="metric-list"><span>Momentary</span><strong>{formatMetric(master.momentary_lufs, 'N/A', ' LUFS')}</strong></div>
            <div className="metric-list"><span>LRA</span><strong>{formatMetric(master.lra_lu, 'N/A', ' LU')}</strong></div>
          </MetricCard>
          <MetricCard title="Mix Balance">
            <div className="balance-donut" />
            <div className="balance-legend">
              <span>Total {channelSummary.total_channels ?? channels.length}</span>
              <span>Active {channelSummary.active_channels ?? 0}</span>
              <span>Named {channelSummary.named_channels ?? 0}</span>
            </div>
          </MetricCard>
          <MetricCard title="AI Confidence">
            <DonutScore value={aiConfidence} label={decisionQueue?.agent_runtime_available === true ? 'High' : 'Offline'} />
            <SpectrumChart compact />
          </MetricCard>
        </div>

        <div className="console-tabs">
          <span className="active">Channels</span><span>Overview</span><span>EQ Match</span><span>Dynamics</span><span>Sends</span><span>Main Bus</span>
          <div className="console-view-tools"><small>View</small><button>▤</button><button>▦</button><button>▥</button><button>⚙</button></div>
        </div>
        <div className="console-strips">
          {channels.slice(0, 14).map((channel) => (
            <ChannelStrip channel={channel} selected={channel.name === 'Lead Vocal' || channel.group === 'VOCALS'} key={channel.id} />
          ))}
        </div>

        <div className="lower-metrics-grid">
          <MetricCard title="Phase Scope"><div className="phase-scope" /></MetricCard>
          <MetricCard title="Correlation"><div className="correlation-value">+0.28</div><MeterBar value={-12} /></MetricCard>
          <MetricCard title="Spectral Balance"><div className="bar-chart">{[40, 78, 66, 58, 82, 39, 73].map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}</div></MetricCard>
          <MetricCard title="Dynamics Overview"><div className="h-bars">{['Drums', 'Bass', 'Guitars', 'Vocals'].map((name, i) => <p key={name}><span>{name}</span><i style={{ width: `${82 - i * 12}%` }} /></p>)}</div></MetricCard>
          <MetricCard title="Loudness History"><SpectrumChart compact /></MetricCard>
          <MetricCard title="AI Suggestions"><div className="big-metric">{decisionSummary.pending_count ?? 0}</div><p>{operatorAnalysisReport?.status || 'pending'}</p></MetricCard>
        </div>
      </div>
      <DirectorPanel decisionQueue={decisionQueue} operatorModeStatus={operatorModeStatus} proposalActions={proposalActions} />
    </div>
  );
}

export function AgentsConsoleView({ decisionQueue, operatorModeStatus, proposalActions, operatorAnalysisReport }) {
  const agentAvailable = decisionQueue?.agent_runtime_available === true;
  const summary = decisionQueue?.summary || {};
  const canApply = operatorModeStatus?.capabilities?.can_apply_to_console === true;
  const agentRows = agentAvailable ? agents : [
    {
      name: 'Mixing Agent Runtime',
      role: 'Очередь предложений и история решений',
      status: 'UNAVAILABLE',
      task: decisionQueue?.reason || 'mixing_agent_runtime_not_initialized',
      confidence: 0,
      tone: 'yellow',
    },
  ];
  return (
    <div className="console-page">
      <div className="console-main">
        <div className="page-heading">
          <div><h2>Консоль агентов</h2><p>Mode: {operatorModeStatus?.label || 'Assist'} · {agentAvailable ? 'runtime connected' : 'runtime unavailable'} · Analysis: {operatorAnalysisReport?.status || 'idle'}</p></div>
          <div className="page-actions">
            <button className="ghost-action" onClick={() => proposalActions?.onImportSafeGain?.()}>SafeGain</button>
            <button className="ghost-action" onClick={() => proposalActions?.onImportSoundcheck?.()}>Soundcheck</button>
            <button className="primary-action" onClick={() => proposalActions?.onAnalyze?.()}>Анализ</button>
          </div>
        </div>
        <div className="agent-summary-grid">
          <MetricCard title="Активные агенты"><div className="big-metric">{agentAvailable ? agentRows.length : 0}</div><p>{agentAvailable ? 'Runtime работает' : 'Не подключён'}</p></MetricCard>
          <MetricCard title="История решений"><div className="big-metric">{summary.history_count ?? 0}</div><p>backend queue</p></MetricCard>
          <MetricCard title="AI Confidence"><div className="big-metric">{agentAvailable ? '87%' : '0%'}</div><p>{agentAvailable ? 'High' : 'Offline'}</p></MetricCard>
          <MetricCard title="Предложений"><div className="big-metric">{summary.pending_count ?? 0}</div><p>Требуют проверки</p></MetricCard>
          <MetricCard title="Применено"><div className="big-metric">{summary.applied_count ?? 0}</div><p>История агента</p></MetricCard>
        </div>
        <div className="console-tabs"><span className="active">Агенты</span><span>Коммуникации</span><span>Очередь задач</span><span>Память</span><span>Производительность</span><span>Настройки</span></div>
        <div className="agent-table">
          <header><span>Агент</span><span>Роль</span><span>Статус</span><span>Текущая задача</span><span>Уверенность</span><span>Активность</span><span>Действия</span></header>
          {agentRows.map((agent) => (
            <div className="agent-row" key={agent.name}>
              <strong>{agent.name}<small>v1.2.{agent.confidence % 10}</small></strong>
              <p>{agent.role}</p>
              <span className={statusClass(agent.status)}>{agent.status}</span>
              <p>{agent.task}</p>
              <div className="confidence-cell"><DonutScore value={agent.confidence} label="" /><span>{agent.confidence >= 85 ? 'High' : 'Medium'}</span></div>
              <MiniSparkline tone={agent.tone} />
              <div className="row-actions"><button>◉</button><button>↗</button><button>⋮</button></div>
            </div>
          ))}
        </div>
      </div>
      <aside className="director-panel">
        <section className="director-card communications-map"><h4>Связи агентов</h4><div className="network-map"><span /><span /><span /><span /><span /></div></section>
        <section className="director-card"><h4>Очередь задач</h4>{(decisionQueue?.pending_actions || []).slice(0, 4).map((task, i) => <div className="task-row task-row-actions" key={task.id || task.title}><b>{task.severity || 'Medium'}</b><p>{task.title}</p><span>{i + 1}</span><div><button onClick={() => proposalActions?.onAccept?.(task.id)}>✓</button>{canApply && task.can_apply && <button onClick={() => proposalActions?.onApply?.(task.id)}>▶</button>}<button onClick={() => proposalActions?.onDismiss?.(task.id)}>×</button></div></div>)}{(!decisionQueue?.pending_actions || decisionQueue.pending_actions.length === 0) && <p className="message-line">{decisionQueue?.reason || 'Очередь пуста'}</p>}</section>
        <section className="director-card"><h4>Сообщения агентов</h4>{decisionsFromQueue(decisionQueue).slice(0, 3).map((item) => <p className="message-line" key={item.join('-')}>{item[1]} · {item[2]}</p>)}</section>
      </aside>
    </div>
  );
}

export function ChannelListView({ inventory }) {
  const channels = channelsFromInventory(inventory);
  const summary = inventory?.summary || {};
  const selectedChannel = channels.find(channel => channel.group === 'VOCALS') || channels[0] || normalizeChannel({});
  return (
    <div className="console-page">
      <div className="console-main">
        <div className="page-heading"><div><h2>Список каналов</h2><p>Все каналы и их текущее состояние</p></div><input className="search-input" placeholder="Поиск канала..." /></div>
        <div className="agent-summary-grid compact">
          <MetricCard title="Всего каналов"><div className="big-metric">{summary.total_channels ?? channels.length}</div></MetricCard>
          <MetricCard title="Активные"><div className="big-metric">{summary.active_channels ?? 0}</div></MetricCard>
          <MetricCard title="Muted"><div className="big-metric warn">{summary.muted_channels ?? 'N/A'}</div></MetricCard>
          <MetricCard title="Inactive"><div className="big-metric purple">{summary.inactive_channels ?? 'N/A'}</div></MetricCard>
          <MetricCard title="Coverage"><div className="big-metric blue">{formatMetric(summary.coverage_percent, '0.0%', '%')}</div></MetricCard>
        </div>
        <div className="channel-table">
          <header><span>#</span><span>Канал</span><span>Тип</span><span>Источник</span><span>Уровень</span><span>LUFS</span><span>Динамика</span><span>Статус</span><span>AI</span><span>Группа</span></header>
          {channels.map((channel) => (
            <div className="channel-row" key={channel.id}>
              <span>{channel.id}</span>
              <strong>{channel.name}</strong>
              <span>{channel.role}</span>
              <span>{channel.source}</span>
              <MeterBar value={channel.level} />
              <span>{formatMetric(channel.lufs)}</span>
              <span>{formatMetric(channel.dynamic, 'N/A', ' dB')}</span>
              <span className={statusClass(channel.status)}>{String(channel.status).toUpperCase()}</span>
              <span className="ai-recognition">{channel.role}{channel.confidence ? ` (${channel.confidence}%)` : ''}</span>
              <span className="group-tag">{channel.group}</span>
            </div>
          ))}
        </div>
      </div>
      <aside className="director-panel channel-detail">
        <section className="director-card"><div className="channel-avatar">MIC</div><h3>{selectedChannel.name}</h3><p>{selectedChannel.source} · {String(selectedChannel.status).toUpperCase()}</p><DonutScore value={selectedChannel.confidence || 0} label="AI" /></section>
        <section className="director-card"><h4>Level Meter</h4><MeterBar value={selectedChannel.level} /><p>Input Level {formatMetric(selectedChannel.level, 'N/A', ' dBFS')}</p><p>LUFS Short {formatMetric(selectedChannel.lufs)}</p><p>True Peak {formatMetric(selectedChannel.peak, 'N/A', ' dBTP')}</p></section>
      </aside>
    </div>
  );
}

export function ConnectionsView({ serverConnected, mixerConnected, topology }) {
  const connection = topology?.connection || {};
  const audioCapture = topology?.audio_capture || {};
  const nodes = topology?.nodes || [];
  const connectedDevices = topology?.connected_devices || [];
  const mixerState = connection.connected ?? mixerConnected;
  const coreNode = nodes.find(node => node.id === 'automixer_core');
  const audioNode = nodes.find(node => node.id === 'audio_capture');
  return (
    <div className="console-page">
      <div className="console-main">
        <div className="console-tabs connection-tabs"><span className="active">System Connection</span><span>Audio Routing</span><span>OSC / MIDI</span><span>Devices</span><span>Network</span><span>Backup & Sync</span></div>
        <div className="topology-board">
          <section className="topology-column"><h4>Клиенты</h4>{['Director Control', 'Engineer Tablet', 'Stage Monitor iPad', 'FOH Laptop'].map((name) => <div className="node-card" key={name}><span>▣</span><div><strong>{name}</strong><p>Online</p></div></div>)}</section>
          <section className="topology-column core"><h4>Automixer Core</h4><div className="node-card large"><span>AM</span><div><strong>Automixer Engine</strong><p>{coreNode?.detail || 'running'}</p></div></div><div className="node-card"><span>AI</span><div><strong>Decision Queue</strong><p>{nodes.find(node => node.id === 'decision_queue')?.detail || 'runtime pending'}</p></div></div></section>
          <section className="topology-column"><h4>Audio Interface</h4><div className="node-card large"><span>W</span><div><strong>{connection.client_type || 'WING RACK'}</strong><p>{mixerState ? 'Connected' : 'Disconnected'}</p><p>{audioCapture.sample_rate ? `${audioCapture.sample_rate / 1000} kHz` : '48 kHz'} · {audioNode?.detail || '0 channel(s)'}</p></div></div></section>
        </div>
        <div className="connection-bottom-grid">
          <section className="director-card">
            <h4>Network Interfaces</h4>
            <div className="status-tile"><b>Control</b><p>{connection.ip || 'No mixer IP'} · {connection.connection_mode || 'offline'}</p></div>
            <div className="status-tile"><b>Ethernet 2</b><p>192.168.2.50 · Standby</p></div>
          </section>
          <section className="director-card">
            <h4>System Latency</h4>
            <div className="status-tile"><b>{audioCapture.running ? 'Running' : 'Offline'}</b><p>Audio capture</p></div>
            <SpectrumChart compact />
          </section>
          <section className="director-card">
            <h4>Sync Status</h4>
            <div className="status-tile"><b>{audioCapture.sample_rate ? 'Known' : 'Unknown'}</b><p>{audioCapture.sample_rate || 'No sample rate'} Hz</p></div>
            <div className="status-tile"><b>{audioCapture.num_channels || 0}</b><p>Audio channels</p></div>
          </section>
        </div>
      </div>
      <aside className="director-panel">
        <section className="director-card"><h4>Connection Status</h4><div className="connection-status-grid"><div className="status-tile"><b>{serverConnected ? 'Good' : 'Offline'}</b><p>Overall Status</p></div><div className="status-tile"><b>{audioCapture.running ? 'Good' : 'Offline'}</b><p>Audio Connection</p></div><div className="status-tile"><b>{mixerState ? 'Good' : 'Safe Offline'}</b><p>Control Connection</p></div><div className="status-tile"><b>{topology ? 'Known' : 'Pending'}</b><p>Topology</p></div></div></section>
        <section className="director-card"><h4>Connected Devices</h4>{connectedDevices.length > 0 ? connectedDevices.map((device) => <p className="message-line" key={`${device.kind}-${device.name}`}>{device.name} · {device.ip || device.channels || ''} · {device.status}</p>) : <p className="message-line">Нет подключённых устройств runtime</p>}</section>
        <section className="director-card"><h4>Connection Log</h4>{(topology?.links || []).map((link) => <p className="message-line" key={`${link.from}-${link.to}`}>{link.kind} · {link.status}</p>)}{!topology?.links?.length && <p className="message-line">Ожидаю topology snapshot</p>}</section>
      </aside>
    </div>
  );
}

export function TransportBar() {
  return (
    <footer className="transport-bar">
      <button className="play-button">▶ PLAY</button>
      <div><strong>00:34:27</strong><span>Elapsed</span></div>
      <div><span>Markers</span><strong>Soundcheck Start</strong></div>
      <div><span>Record / Snapshot</span><button>REC</button><button>Snapshot</button></div>
      <div><span>Automation</span><strong>Ride · On</strong></div>
      <div><span>Safe Limit</span><strong>Medium</strong></div>
      <div><span>Rollback</span><strong>Last action 10:24:15</strong></div>
    </footer>
  );
}
