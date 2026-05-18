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

function DirectorPanel() {
  return (
    <aside className="director-panel">
      <section className="director-card director-profile">
        <div className="director-avatar">AI</div>
        <div>
          <h3>Директор Automixer</h3>
          <p>Главный управляющий агент</p>
          <span className="online-dot">Online</span>
        </div>
      </section>
      <section className="director-card">
        <h4>Текущая цель</h4>
        <p>Обеспечить читаемость вокала и сбалансированный микс без live-записей вне разрешённого режима.</p>
        <button className="ghost-action">Детали цели</button>
      </section>
      <section className="director-card">
        <h4>Последние решения</h4>
        {decisions.map(([time, text, state]) => (
          <div className="decision-row" key={`${time}-${text}`}>
            <span>{time}</span>
            <p>{text}</p>
            <strong>{state}</strong>
          </div>
        ))}
      </section>
      <section className="director-card">
        <h4>Предложения агента</h4>
        <div className="proposal-row">
          <p>Lead Vocal: добавить воздуха +1.5 dB @ 12 kHz</p>
          <div><button>Принять</button><button>Отклонить</button></div>
        </div>
        <div className="proposal-row">
          <p>Kick: убрать мутность -2 dB @ 250 Hz</p>
          <div><button>Принять</button><button>Отклонить</button></div>
        </div>
      </section>
    </aside>
  );
}

function ChannelStrip({ channel, selected }) {
  return (
    <div className={`console-strip ${selected ? 'selected' : ''}`}>
      <div className="strip-head">
        <strong>{channel.name}</strong>
        <span>{channel.role}</span>
      </div>
      <b className="strip-db">{channel.level.toFixed(1)} dB</b>
      <MiniSparkline tone={channel.group === 'VOCALS' ? 'purple' : 'green'} />
      <div className="strip-tools"><span>GATE</span><span>COMP</span></div>
      <div className="strip-fader">
        <div className="strip-meter"><span style={{ height: `${Math.max(12, Math.min(92, 80 + channel.level))}%` }} /></div>
        <div className="fader-rail"><i style={{ bottom: `${Math.max(8, Math.min(84, 62 + channel.level))}%` }} /></div>
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

export function DashboardView() {
  return (
    <div className="console-page console-page-dashboard">
      <div className="console-main">
        <div className="metrics-grid">
          <MetricCard title="Master Bus">
            <div className="master-values">
              <div className="big-metric">-14.2 <span>LUFS</span></div>
              <div className="big-metric small">-1.1 <span>dBTP</span></div>
            </div>
            <div className="master-labels"><span>Integrated</span><span>True Peak</span></div>
            <MeterBar value={-14.2} />
          </MetricCard>
          <MetricCard title="Spectrum">
            <SpectrumChart />
            <div className="spectrum-axis"><span>20</span><span>100</span><span>1k</span><span>10k</span></div>
          </MetricCard>
          <MetricCard title="Loudness">
            <div className="metric-list"><span>Short Term</span><strong>-13.8 LUFS</strong></div>
            <div className="metric-list"><span>Momentary</span><strong>-12.1 LUFS</strong></div>
            <div className="metric-list"><span>LRA</span><strong>6.7 LU</strong></div>
          </MetricCard>
          <MetricCard title="Mix Balance">
            <div className="balance-donut" />
            <div className="balance-legend"><span>Drums 28%</span><span>Vocals 26%</span><span>Bass 12%</span></div>
          </MetricCard>
          <MetricCard title="AI Confidence">
            <DonutScore value={87} label="High" />
            <SpectrumChart compact />
          </MetricCard>
        </div>

        <div className="console-tabs">
          <span className="active">Channels</span><span>Overview</span><span>EQ Match</span><span>Dynamics</span><span>Sends</span><span>Main Bus</span>
          <div className="console-view-tools"><small>View</small><button>▤</button><button>▦</button><button>▥</button><button>⚙</button></div>
        </div>
        <div className="console-strips">
          {mixChannels.slice(0, 14).map((channel) => (
            <ChannelStrip channel={channel} selected={channel.name === 'Lead Vocal'} key={channel.id} />
          ))}
        </div>

        <div className="lower-metrics-grid">
          <MetricCard title="Phase Scope"><div className="phase-scope" /></MetricCard>
          <MetricCard title="Correlation"><div className="correlation-value">+0.28</div><MeterBar value={-12} /></MetricCard>
          <MetricCard title="Spectral Balance"><div className="bar-chart">{[40, 78, 66, 58, 82, 39, 73].map((h, i) => <i key={i} style={{ height: `${h}%` }} />)}</div></MetricCard>
          <MetricCard title="Dynamics Overview"><div className="h-bars">{['Drums', 'Bass', 'Guitars', 'Vocals'].map((name, i) => <p key={name}><span>{name}</span><i style={{ width: `${82 - i * 12}%` }} /></p>)}</div></MetricCard>
          <MetricCard title="Loudness History"><SpectrumChart compact /></MetricCard>
          <MetricCard title="AI Suggestions"><MiniSparkline tone="dots" /></MetricCard>
        </div>
      </div>
      <DirectorPanel />
    </div>
  );
}

export function AgentsConsoleView() {
  return (
    <div className="console-page">
      <div className="console-main">
        <div className="page-heading">
          <div><h2>Консоль агентов</h2><p>Управление AI-агентами Automixer</p></div>
          <button className="primary-action">+ Создать агента</button>
        </div>
        <div className="agent-summary-grid">
          <MetricCard title="Активные агенты"><div className="big-metric">8</div><p>Все системы работают</p></MetricCard>
          <MetricCard title="Решения сегодня"><div className="big-metric">142</div><p>28% от вчера</p></MetricCard>
          <MetricCard title="AI Confidence"><div className="big-metric">87%</div><p>High</p></MetricCard>
          <MetricCard title="Предложений"><div className="big-metric">23</div><p>Требуют проверки</p></MetricCard>
          <MetricCard title="Применено"><div className="big-metric">12</div><p>За последние 30 мин</p></MetricCard>
        </div>
        <div className="console-tabs"><span className="active">Агенты</span><span>Коммуникации</span><span>Очередь задач</span><span>Память</span><span>Производительность</span><span>Настройки</span></div>
        <div className="agent-table">
          <header><span>Агент</span><span>Роль</span><span>Статус</span><span>Текущая задача</span><span>Уверенность</span><span>Активность</span><span>Действия</span></header>
          {agents.map((agent) => (
            <div className="agent-row" key={agent.name}>
              <strong>{agent.name}<small>v1.2.{agent.confidence % 10}</small></strong>
              <p>{agent.role}</p>
              <span className="state-green">{agent.status}</span>
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
        <section className="director-card"><h4>Очередь задач</h4>{['Оценить баланс барабанов', 'Lead Vocal: проверить компрессию', 'Bass: анализ низких частот', 'Гитара 1: проверить резонансы'].map((task, i) => <div className="task-row" key={task}><b>{i < 2 ? 'High' : 'Medium'}</b><p>{task}</p><span>10:{24 + i}</span></div>)}</section>
        <section className="director-card"><h4>Сообщения агентов</h4>{['Decision Agent -> Safety Agent', 'Safety Agent -> Decision Agent', 'Mix Critic Agent -> All Agents'].map((item) => <p className="message-line" key={item}>{item}</p>)}</section>
      </aside>
    </div>
  );
}

export function ChannelListView() {
  return (
    <div className="console-page">
      <div className="console-main">
        <div className="page-heading"><div><h2>Список каналов</h2><p>Все каналы и их текущее состояние</p></div><input className="search-input" placeholder="Поиск канала..." /></div>
        <div className="agent-summary-grid compact">
          <MetricCard title="Всего каналов"><div className="big-metric">48</div></MetricCard>
          <MetricCard title="Активные"><div className="big-metric">36</div></MetricCard>
          <MetricCard title="Muted"><div className="big-metric warn">6</div></MetricCard>
          <MetricCard title="Inactive"><div className="big-metric purple">6</div></MetricCard>
          <MetricCard title="Coverage"><div className="big-metric blue">98%</div></MetricCard>
        </div>
        <div className="channel-table">
          <header><span>#</span><span>Канал</span><span>Тип</span><span>Источник</span><span>Уровень</span><span>LUFS</span><span>Динамика</span><span>Статус</span><span>AI</span><span>Группа</span></header>
          {mixChannels.map((channel) => (
            <div className="channel-row" key={channel.id}>
              <span>{channel.id}</span>
              <strong>{channel.name}</strong>
              <span>{channel.role}</span>
              <span>In {String(channel.id).padStart(2, '0')}</span>
              <MeterBar value={channel.level} />
              <span>{channel.lufs}</span>
              <span>{channel.dynamic} dB</span>
              <span className="state-green">ACTIVE</span>
              <span className="ai-recognition">{channel.role} ({channel.confidence}%)</span>
              <span className="group-tag">{channel.group}</span>
            </div>
          ))}
        </div>
      </div>
      <aside className="director-panel channel-detail">
        <section className="director-card"><div className="channel-avatar">MIC</div><h3>Lead Vocal</h3><p>In 12 · ACTIVE</p><DonutScore value={99} label="AI" /></section>
        <section className="director-card"><h4>Level Meter</h4><MeterBar value={-9.3} /><p>Input Level -9.3 dBFS</p><p>LUFS Short -10.2</p><p>True Peak -1.1 dBTP</p></section>
      </aside>
    </div>
  );
}

export function ConnectionsView({ serverConnected, mixerConnected }) {
  return (
    <div className="console-page">
      <div className="console-main">
        <div className="console-tabs connection-tabs"><span className="active">System Connection</span><span>Audio Routing</span><span>OSC / MIDI</span><span>Devices</span><span>Network</span><span>Backup & Sync</span></div>
        <div className="topology-board">
          <section className="topology-column"><h4>Клиенты</h4>{['Director Control', 'Engineer Tablet', 'Stage Monitor iPad', 'FOH Laptop'].map((name) => <div className="node-card" key={name}><span>▣</span><div><strong>{name}</strong><p>Online</p></div></div>)}</section>
          <section className="topology-column core"><h4>Automixer Core</h4><div className="node-card large"><span>AM</span><div><strong>Automixer Engine</strong><p>v0.8.7 · running</p></div></div><div className="node-card"><span>AI</span><div><strong>AI Processing</strong><p>8 agents active</p></div></div></section>
          <section className="topology-column"><h4>Audio Interface</h4><div className="node-card large"><span>W</span><div><strong>WING RACK</strong><p>{mixerConnected ? 'Connected' : 'Disconnected'}</p><p>48 kHz · 0.8 ms</p></div></div></section>
        </div>
        <div className="connection-bottom-grid">
          <section className="director-card">
            <h4>Network Interfaces</h4>
            <div className="status-tile"><b>Ethernet 1</b><p>192.168.1.50 · 1 Gbps · Up</p></div>
            <div className="status-tile"><b>Ethernet 2</b><p>192.168.2.50 · Standby</p></div>
          </section>
          <section className="director-card">
            <h4>System Latency</h4>
            <div className="status-tile"><b>0.8 ms</b><p>Audio round trip</p></div>
            <SpectrumChart compact />
          </section>
          <section className="director-card">
            <h4>Sync Status</h4>
            <div className="status-tile"><b>Locked</b><p>48 kHz · word clock</p></div>
            <div className="status-tile"><b>0.8 ms</b><p>Latency compensation</p></div>
          </section>
        </div>
      </div>
      <aside className="director-panel">
        <section className="director-card"><h4>Connection Status</h4><div className="connection-status-grid"><div className="status-tile"><b>{serverConnected ? 'Good' : 'Offline'}</b><p>Overall Status</p></div><div className="status-tile"><b>Good</b><p>Audio Connection</p></div><div className="status-tile"><b>{mixerConnected ? 'Good' : 'Safe Offline'}</b><p>Control Connection</p></div><div className="status-tile"><b>Good</b><p>Network Status</p></div></div></section>
        <section className="director-card"><h4>Connected Devices</h4>{['WING RACK · 192.168.1.10', 'Stage Box 1 · 192.168.1.11', 'Stage Box 2 · 192.168.1.12', 'Powerplay P16 · 192.168.1.13'].map((line) => <p className="message-line" key={line}>{line} · Online</p>)}</section>
        <section className="director-card"><h4>Connection Log</h4>{['Connected to WING RACK', 'OSC connection established', 'Audio stream started', 'All systems ready'].map((line, i) => <p className="message-line" key={line}>10:2{i} · {line}</p>)}</section>
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
