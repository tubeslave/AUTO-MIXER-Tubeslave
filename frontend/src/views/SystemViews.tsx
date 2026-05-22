import type { CSSProperties } from 'react';
import {
  ActionCluster,
  DataTable,
  LevelMeter,
  MetricCard,
  Panel,
  Sparkline,
  StatusBadge
} from '../components/prototype/AutomixerComponents';
import { channels, deviceHealth, devices, logs } from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { DeviceRecord, LogRecord, Metric, RoutingSegment, Tone, TopologyNode, UiActions, UiState } from '../types';
import {
  logKey,
  openActionFocus,
  runtimeRoutingSegments,
  runtimeTopologyLinks,
  runtimeTopologyNodes,
  SettingRows,
  statusToneForDevice,
  statusToneForLog,
  tableView,
  TechnicalChecklist
} from './viewHelpers';

const topologyLayout: Record<string, { left: number; top: number; width: number; height: number }> = {
  director_control: { left: 4, top: 18, width: 23, height: 42 },
  automixer_core: { left: 38, top: 15, width: 26, height: 38 },
  mixer: { left: 74, top: 22, width: 20, height: 30 },
  audio_capture: { left: 38, top: 70, width: 20, height: 18 },
  decision_queue: { left: 9, top: 70, width: 21, height: 18 }
};

function topologyTone(node: TopologyNode): Tone {
  if (node.status !== 'online' && node.status !== 'running' && node.status !== 'connected') return 'yellow';
  if (node.kind === 'core' || node.kind === 'agent') return 'purple';
  if (node.kind === 'audio' || node.kind === 'client') return 'blue';
  return 'green';
}

function linkStroke(kind: string) {
  if (kind.includes('audio')) return '#a371f7';
  if (kind.includes('osc') || kind.includes('websocket')) return '#00d4ff';
  if (kind.includes('proposal')) return '#d29922';
  return '#2dd36f';
}

function nodeBox(node: TopologyNode, index: number) {
  return topologyLayout[node.id] ?? {
    left: 8 + (index % 4) * 22,
    top: 18 + Math.floor(index / 4) * 26,
    width: 19,
    height: 19
  };
}

function boxCenter(box: { left: number; top: number; width: number; height: number }) {
  return {
    x: box.left + box.width / 2,
    y: box.top + box.height / 2
  };
}

function TopologyBox({
  title,
  lines,
  tone = 'green',
  selected = false,
  className = '',
  style,
  onClick
}: {
  title: string;
  lines: string[];
  tone?: Tone;
  selected?: boolean;
  className?: string;
  style?: CSSProperties;
  onClick: () => void;
}) {
  const toneClass = tone === 'blue'
    ? 'border-audio/45 text-audio'
    : tone === 'purple'
      ? 'border-ai/45 text-ai'
      : tone === 'yellow'
        ? 'border-warn/45 text-warn'
        : tone === 'red'
          ? 'border-danger/45 text-danger'
          : 'border-online/45 text-online';

  return (
    <button
      className={`absolute rounded-[7px] border bg-[#0d151e]/95 p-2 text-left shadow-[0_0_18px_rgba(0,0,0,.35)] transition hover:border-audio ${toneClass} ${selected ? 'ring-1 ring-audio' : ''} ${className}`}
      onClick={onClick}
      style={style}
      type="button"
    >
      <div className="mb-1 text-center text-[10px] font-black uppercase tracking-[0.08em] text-text">{title}</div>
      <div className="space-y-1">
        {lines.map(line => (
          <div className="flex items-center gap-1.5 text-[9px]" key={line}>
            <i className="h-1.5 w-1.5 rounded-full bg-current" />
            <span className="truncate text-muted">{line}</span>
          </div>
        ))}
      </div>
    </button>
  );
}

function SystemTopologyDiagram({
  nodes,
  links,
  selectedNodeId,
  onSelect
}: {
  nodes: TopologyNode[];
  links: ReturnType<typeof runtimeTopologyLinks>;
  selectedNodeId: string;
  onSelect: (id: string) => void;
}) {
  const nodeById = new Map(nodes.map(node => [node.id, node]));

  return (
    <div className="relative h-full min-h-[300px] overflow-hidden rounded-[8px] border border-line bg-[#091018]">
      <div className="absolute right-3 top-3 z-10 flex gap-2">
        <button className="am-button" type="button">Auto Layout</button>
        <button className="am-button" type="button">Refresh</button>
      </div>
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <filter id="topology-glow"><feGaussianBlur stdDeviation="1.2" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {links.map((link, index) => {
          const sourceNode = nodeById.get(link.from);
          const targetNode = nodeById.get(link.to);
          if (!sourceNode || !targetNode) return null;
          const source = boxCenter(nodeBox(sourceNode, nodes.indexOf(sourceNode)));
          const target = boxCenter(nodeBox(targetNode, nodes.indexOf(targetNode)));
          return (
            <path
              d={`M ${source.x} ${source.y} C ${(source.x + target.x) / 2} ${source.y}, ${(source.x + target.x) / 2} ${target.y}, ${target.x} ${target.y}`}
              fill="none"
              filter="url(#topology-glow)"
              key={`${link.from}-${link.to}-${index}`}
              stroke={linkStroke(link.kind)}
              strokeDasharray={link.status === 'online' ? undefined : '1.4 1'}
              strokeWidth="0.7"
            />
          );
        })}
      </svg>
      {nodes.map((node, index) => {
        const box = nodeBox(node, index);
        return (
          <TopologyBox
            className="min-w-0"
            key={node.id}
            lines={[`${node.kind} · ${node.status}`, node.detail]}
            onClick={() => onSelect(node.id)}
            selected={selectedNodeId === node.id}
            style={{ left: `${box.left}%`, top: `${box.top}%`, width: `${box.width}%`, height: `${box.height}%` }}
            title={node.label}
            tone={topologyTone(node)}
          />
        );
      })}
      <div className="absolute bottom-2 left-3 flex gap-5 text-[9px] text-muted">
        <span><i className="mr-1 inline-block h-1 w-5 bg-ai" />Audio</span>
        <span><i className="mr-1 inline-block h-1 w-5 bg-audio" />OSC / Control</span>
        <span><i className="mr-1 inline-block h-1 w-5 bg-online" />Internal</span>
        <span><i className="mr-1 inline-block h-1 w-5 bg-warn" />Network</span>
      </div>
    </div>
  );
}

function TopologyInspector({
  node,
  nodes,
  data
}: {
  node: TopologyNode;
  nodes: TopologyNode[];
  data: OperatorViewData;
}) {
  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Connection Status">
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="truncate text-[15px] font-bold text-text">{node.label}</h3>
          <StatusBadge tone={topologyTone(node)}>{node.status}</StatusBadge>
        </div>
        <TechnicalChecklist rows={[
          { label: 'Kind', value: node.kind, tone: topologyTone(node) },
          { label: 'Detail', value: node.detail, tone: 'blue' },
          { label: 'Nodes', value: String(nodes.length), tone: 'purple' },
          { label: 'Writes', value: 'Read-only', tone: 'red' }
        ]} />
      </Panel>
      <Panel title="Connected Devices" className="flex-1">
        {(data.devices.length ? data.devices : devices).slice(0, 7).map(device => (
          <div className="mb-3 grid grid-cols-[1fr_68px_48px] gap-2 text-[10px]" key={device.name}>
            <span className="truncate text-text">{device.name}</span>
            <span className="font-mono text-muted">{device.sampleRate}</span>
            <span className={statusToneForDevice(device.status) === 'green' ? 'text-online' : 'text-warn'}>{device.status}</span>
          </div>
        ))}
      </Panel>
      <Panel title="Connection Log">
        {(data.logs.length ? data.logs : logs).slice(0, 5).map(log => <p className="mb-1 truncate text-[10px] text-muted" key={`${log.time}-${log.source}`}>{log.time} · {log.message}</p>)}
      </Panel>
    </div>
  );
}

type RoutingListRow = { id: string; label: string };

function RoutingList({
  title,
  rows,
  tone = 'green',
  selectedId,
  onSelect
}: {
  title: string;
  rows: RoutingListRow[];
  tone?: Tone;
  selectedId?: string;
  onSelect?: (id: string) => void;
}) {
  const color = tone === 'blue' ? 'text-audio' : tone === 'purple' ? 'text-ai' : tone === 'yellow' ? 'text-warn' : tone === 'red' ? 'text-danger' : 'text-online';
  return (
    <div className="relative z-10 rounded-[7px] border border-line bg-[#0c141d]/95 p-2">
      <div className="am-label mb-2">{title}</div>
      <div className="space-y-1">
        {rows.map((row, index) => {
          const selected = selectedId === row.id;
          const content = (
            <>
              <span className={`font-mono ${color}`}>{String(index + 1).padStart(2, '0')}</span>
              <span className="truncate text-muted">{row.label}</span>
            </>
          );
          return onSelect ? (
            <button
              className={`grid w-full grid-cols-[28px_1fr] items-center gap-2 rounded border px-2 py-1 text-left text-[9px] transition ${selected ? 'border-audio bg-audio/10' : 'border-line/80 bg-row hover:border-audio'}`}
              key={row.id}
              onClick={() => onSelect(row.id)}
              type="button"
            >
              {content}
            </button>
          ) : (
            <div className="grid grid-cols-[28px_1fr] items-center gap-2 rounded border border-line/80 bg-row px-2 py-1 text-[9px]" key={row.id}>
              {content}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function AudioRoutingMatrix({
  data,
  routingRows,
  selectedSegmentId,
  onSelectSegment
}: {
  data: OperatorViewData;
  routingRows: RoutingSegment[];
  selectedSegmentId: string;
  onSelectSegment: (id: string) => void;
}) {
  const inputRows = (data.channels.length ? data.channels : channels)
    .slice(0, 16)
    .map(channel => ({ id: `input-${channel.id}`, label: `${channel.source} ${channel.name}` }));
  const segmentRows = routingRows.slice(0, 10).map(segment => ({ id: segment.id, label: segment.labelShort }));
  const outputRows = routingRows.length
    ? routingRows.slice(0, 10).map(segment => {
      const requirement = segment.required ? 'REQ' : 'OPT';
      return { id: `output-${segment.id}`, label: `${requirement} · ${segment.tapPoint || segment.role}` };
    })
    : ['Out 01 Main L', 'Out 02 Main R', 'Out 03 Monitor A', 'Out 04 Monitor B', 'Out 05 IEM L', 'Out 06 IEM R', 'Out 07 Broadcast L', 'Out 08 Broadcast R']
      .map((label, index) => ({ id: `out-${index}`, label }));
  const colors = ['#00d4ff', '#a371f7', '#2dd36f', '#d29922', '#f85149'];
  const segmentCount = Math.max(segmentRows.length, 1);

  return (
    <div className="relative h-full min-h-[410px] overflow-hidden rounded-[8px] border border-line bg-[#091018] p-3">
      <svg className="pointer-events-none absolute inset-0 h-full w-full opacity-80" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          <filter id="routing-glow"><feGaussianBlur stdDeviation="1.2" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        {inputRows.slice(0, 12).map((_, index) => {
          const y = 14 + index * 5.6;
          const target = 20 + (index % segmentCount) * 7.5;
          return <path d={`M 26 ${y} C 39 ${y}, 36 ${target}, 50 ${target}`} fill="none" filter="url(#routing-glow)" key={`in-${index}`} stroke={colors[index % colors.length]} strokeWidth="0.35" />;
        })}
        {outputRows.map((_, index) => {
          const y = 19 + index * 8.2;
          const source = 21 + (index % segmentCount) * 7.5;
          return <path d={`M 60 ${source} C 73 ${source}, 70 ${y}, 82 ${y}`} fill="none" filter="url(#routing-glow)" key={`out-${index}`} stroke={colors[(index + 2) % colors.length]} strokeWidth="0.35" />;
        })}
      </svg>
      <div className="grid h-full grid-cols-[30%_26%_30%] justify-between gap-6">
        <RoutingList rows={inputRows} title={`Inputs (${inputRows.length})`} tone="blue" />
        <RoutingList onSelect={onSelectSegment} rows={segmentRows} selectedId={selectedSegmentId} title="Dante / Tap Scheme" tone="purple" />
        <RoutingList rows={outputRows} title="Tap / Requirement" tone="green" />
      </div>
    </div>
  );
}

function RoutingInspector({ segment, sourceLabel }: { segment: RoutingSegment; sourceLabel: string }) {
  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="Routing Segment">
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="truncate text-[15px] font-bold text-text">{segment.labelShort}</h3>
          <StatusBadge tone={segment.required ? 'yellow' : 'blue'}>{segment.required ? 'Required' : 'Optional'}</StatusBadge>
        </div>
        <TechnicalChecklist rows={[
          { label: 'Source', value: sourceLabel, tone: sourceLabel === 'Runtime' ? 'green' : 'blue' },
          { label: 'Channels', value: `${segment.start}-${segment.end}`, tone: 'blue' },
          { label: 'Role', value: segment.role, tone: 'purple' },
          { label: 'Tap point', value: segment.tapPoint, tone: 'green' },
          { label: 'Stereo', value: segment.stereo ? 'Yes' : 'No', tone: segment.stereo ? 'green' : 'yellow' }
        ]} />
      </Panel>
      <Panel title="Wing Hint">
        <p className="text-[11px] leading-5 text-muted">{segment.wingRoutingHint}</p>
      </Panel>
      <Panel title="Used By" className="flex-1">
        {(segment.usedBy.length ? segment.usedBy : ['No assigned consumer']).map(item => (
          <div className="mb-2 rounded-[5px] border border-line bg-row px-2 py-1 text-[10px] text-muted" key={item}>{item}</div>
        ))}
        <div className="mt-4">
          <TechnicalChecklist rows={[
            { label: 'Routing writes', value: 'Disabled', tone: 'red' },
            { label: 'Inspect only', value: 'Enabled', tone: 'green' }
          ]} />
        </div>
      </Panel>
    </div>
  );
}

type OscRow = LogRecord & { key: string };

function OscMessageInspector({ row, ui }: { row: OscRow; ui: UiState }) {
  const connectionTone: Tone = ui.backend.connectionState === 'connected' ? 'green' : ui.backend.connectionState === 'connecting' ? 'yellow' : ui.backend.connectionState === 'error' ? 'red' : 'blue';
  return (
    <div className="flex h-full flex-col gap-2">
      <Panel title="OSC / MIDI Message">
        <div className="mb-3 flex items-center justify-between gap-2">
          <h3 className="truncate text-[15px] font-bold text-text">{row.source}</h3>
          <StatusBadge tone={statusToneForLog(row.level)}>{row.level}</StatusBadge>
        </div>
        <p className="mb-4 text-[11px] leading-5 text-muted">{row.message}</p>
        <TechnicalChecklist rows={[
          { label: 'Time', value: row.time, tone: 'blue' },
          { label: 'Transport', value: ui.backend.connectionState, tone: connectionTone },
          { label: 'Apply path', value: ui.backend.applyPath, tone: ui.backend.canApplyToConsole ? 'yellow' : 'blue' },
          { label: 'Direct OSC writes', value: 'Disabled', tone: 'red' }
        ]} />
      </Panel>
      <Panel title="Read-only Transport" className="flex-1">
        <TechnicalChecklist rows={[
          { label: 'Product Layer', value: ui.backend.dataStateLabel, tone: ui.backend.dataStateTone },
          { label: 'Target', value: ui.backend.connectionLabel, tone: connectionTone },
          { label: 'Snapshot writes', value: 'Disabled', tone: 'red' },
          { label: 'Routing writes', value: 'Disabled', tone: 'red' }
        ]} />
      </Panel>
    </div>
  );
}

function DeviceInspector({ device }: { device: DeviceRecord }) {
  return (
    <Panel title="Selected Device" className="h-full">
      <h3 className="text-lg font-bold">{device.name}</h3>
      <p className="mt-2 text-[11px] leading-5 text-muted">
        {device.type}<br />
        IP: {device.ip}<br />
        Sample Rate: {device.sampleRate}<br />
        I/O: {device.io}
      </p>
      <div className="mt-4"><StatusBadge tone={statusToneForDevice(device.status)}>{device.status}</StatusBadge></div>
      <div className="mt-4 h-24"><Sparkline tone="green" filled /></div>
      <button className="am-button mt-4" type="button">Inspect</button>
    </Panel>
  );
}

function LogInspector({ log }: { log: LogRecord }) {
  return (
    <Panel title="Log Details" className="h-full">
      <div className="mb-3 flex items-center justify-between">
        <StatusBadge tone={statusToneForLog(log.level)}>{log.level}</StatusBadge>
        <span className="font-mono text-[10px] text-muted">{log.time}</span>
      </div>
      <h3 className="text-base font-bold">{log.source}</h3>
      <p className="mt-2 text-[11px] leading-5 text-muted">{log.message}</p>
      <p className="mt-4 text-[11px] leading-5 text-muted">This is read-only frontend/backend event data. No live command was executed.</p>
    </Panel>
  );
}

export function SystemConnection(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const nodes = runtimeTopologyNodes(data.topologyNodes);
  const links = runtimeTopologyLinks(data.topologyLinks);
  const selectedNode = nodes.find(node => node.id === ui.selectedTopologyNodeId) ?? nodes.find(node => node.id === 'automixer_core') ?? nodes[0];
  return tableView('10. System Connection', 'Live topology, latency, and connection status', (
    <div className="grid h-full grid-rows-[1fr_100px] gap-2">
      <Panel title="System Topology"><SystemTopologyDiagram links={links} nodes={nodes} onSelect={actions.selectTopologyNode} selectedNodeId={selectedNode.id} /></Panel>
      <div className="grid grid-cols-3 gap-2">
        <Panel title="Network Interfaces"><SettingRows rows={['Ethernet 1 1Gbps Up', 'Ethernet 2 1Gbps Standby']} /></Panel>
        <Panel title="System Latency"><Sparkline tone="green" filled /></Panel>
        <Panel title="Sync Status"><SettingRows rows={['System Clock Locked', 'Sample Rate 48 kHz', 'Latency Compensation 0.8 ms']} /></Panel>
      </div>
    </div>
  ), <TopologyInspector data={data} node={selectedNode} nodes={nodes} />);
}

export function AudioRouting(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const routingRows = runtimeRoutingSegments(data);
  const selectedSegment = routingRows.find(segment => segment.id === ui.selectedRoutingSegmentId) ?? routingRows[0];
  return tableView('11. Audio Routing', 'Inputs, channel buses, and output map', (
    <AudioRoutingMatrix
      data={data}
      onSelectSegment={actions.selectRoutingSegment}
      routingRows={routingRows}
      selectedSegmentId={selectedSegment.id}
    />
  ), <RoutingInspector segment={selectedSegment} sourceLabel={data.routingSegments.length ? 'Runtime' : 'Mock'} />);
}

export function OscMidi(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const sourceRows = data.logs.length ? data.logs.slice(0, 24) : logs;
  const oscRows = sourceRows.map((log, index) => ({ ...log, key: `${log.time}-${log.source}-${log.message}-${index}` }));
  const selectedRow = oscRows.find(row => row.key === ui.selectedOscMessageKey) ?? oscRows[0];
  const deviceRows = data.devices.length ? data.devices : devices.slice(0, 3);

  return tableView('12. OSC / MIDI', 'OSC devices, message stream, and control settings', (
    <div className="grid h-full grid-cols-[0.8fr_1.4fr_0.8fr] gap-2">
      <Panel title="Runtime Devices">
        <div className="space-y-2">
          {deviceRows.map(device => (
            <div className="rounded-[6px] border border-line bg-row p-2 text-[10px]" key={`${device.name}-${device.ip}`}>
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="truncate font-bold text-text">{device.name}</span>
                <StatusBadge tone={statusToneForDevice(device.status)}>{device.status}</StatusBadge>
              </div>
              <div className="grid grid-cols-[48px_1fr] gap-y-1 text-muted">
                <span>IP</span><span className="truncate text-text">{device.ip}</span>
                <span>Type</span><span className="truncate text-text">{device.type}</span>
                <span>I/O</span><span className="truncate text-text">{device.io}</span>
              </div>
            </div>
          ))}
        </div>
      </Panel>
      <DataTable
        columns={[{ key: 'time', label: 'Time' }, { key: 'source', label: 'Source' }, { key: 'message', label: 'Event / Address' }, { key: 'level', label: 'Kind', render: row => <StatusBadge tone={statusToneForLog(row.level)}>{row.level}</StatusBadge> }]}
        onSelect={row => actions.selectOscMessage(row.key)}
        rowKey={row => row.key}
        rows={oscRows}
        searchable
        searchPlaceholder="Search OSC/MIDI event..."
        selectedRow={selectedRow}
      />
      <OscMessageInspector row={selectedRow} ui={ui} />
    </div>
  ), <OscMessageInspector row={selectedRow} ui={ui} />);
}

export function Devices(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const deviceRows = data.devices.length ? data.devices : devices;
  const selectedDevice = deviceRows.find(device => device.name === ui.selectedDeviceName) ?? deviceRows[0] ?? devices[0];
  return tableView('13. Devices', 'Audio devices and controller inventory', (
    <div className="grid h-full grid-rows-[76px_1fr_108px] gap-2">
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Connected', value: String(deviceRows.length), sub: ui.backend.sourceLabel, tone: 'green' },
          { label: 'Audio I/O', value: '80 ch', sub: 'inputs + outputs', tone: 'blue' },
          { label: 'Controllers', value: '2', sub: 'director surfaces', tone: 'purple' },
          { label: 'Warnings', value: '0', sub: 'prototype health', tone: 'green' }
        ].map(metric => <MetricCard metric={metric as Metric} key={metric.label} />)}
      </div>
      <DataTable
        columns={[
          { key: 'name', label: 'Device', width: '1.2fr' },
          { key: 'ip', label: 'IP', width: '0.9fr' },
          { key: 'type', label: 'Type', width: '1fr' },
          { key: 'sampleRate', label: 'Sample Rate', width: '0.8fr' },
          { key: 'io', label: 'I/O', width: '1.1fr' },
          { key: 'status', label: 'Status', width: '0.75fr', render: row => <StatusBadge tone={statusToneForDevice(row.status)}>{row.status}</StatusBadge> },
          { key: 'actions', label: 'Actions', width: '86px', render: row => <ActionCluster actions={['Inspect', 'Meter', 'More']} onAction={action => {
            actions.selectDevice(row.name);
            openActionFocus(actions, action, row.name, `${row.type} at ${row.ip}. Device inspection is mock/read-only.`, row.type.includes('Console') ? 'green' : 'blue');
          }} /> }
        ]}
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Console', predicate: device => device.type.includes('Console') },
          { label: 'I/O', predicate: device => device.type.includes('I/O') },
          { label: 'Online', predicate: device => device.status === 'Online' }
        ]}
        getSearchText={device => `${device.name} ${device.ip} ${device.type} ${device.io} ${device.status}`}
        onSelect={device => actions.selectDevice(device.name)}
        rowKey={device => device.name}
        rows={deviceRows}
        searchable
        searchPlaceholder="Search device..."
        selectedRow={selectedDevice}
      />
      <div className="grid grid-cols-4 gap-2">
        {deviceHealth.map(item => (
          <Panel title={item.name} key={item.name}>
            <div className="flex items-center justify-between text-[11px]"><StatusBadge tone={item.tone as Tone}>{item.value}</StatusBadge><span className="font-mono text-muted">{item.latency}</span></div>
            <div className="mt-4 h-8"><Sparkline tone={item.tone as Tone} filled /></div>
          </Panel>
        ))}
      </div>
    </div>
  ), <DeviceInspector device={selectedDevice} />);
}

export function Logs(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const sourceRows = data.logs.length ? data.logs : logs;
  const logRows = sourceRows.map((log, index) => ({ ...log, key: logKey(log, index) }));
  const selectedLog = logRows.find(log => log.key === ui.selectedLogKey) ?? logRows[0];
  return tableView('14. Logs', 'System event log and audit trail', (
    <div className="grid h-full grid-rows-[76px_1fr] gap-2">
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Events', value: String(logRows.length), sub: ui.backend.sourceLabel, tone: 'blue' },
          { label: 'Warnings', value: String(logRows.filter(log => log.level === 'WARN').length), sub: 'read-only stream', tone: 'yellow' },
          { label: 'Errors', value: String(logRows.filter(log => log.level === 'ERROR').length), sub: 'runtime issues', tone: logRows.some(log => log.level === 'ERROR') ? 'red' : 'green' },
          { label: 'Success', value: String(logRows.filter(log => log.level === 'SUCCESS').length), sub: ui.backend.statusLabel, tone: 'green' }
        ].map(metric => <MetricCard metric={metric as Metric} key={metric.label} />)}
      </div>
      <DataTable
        columns={[
          { key: 'time', label: 'Time', width: '0.8fr' },
          { key: 'level', label: 'Level', width: '0.75fr', render: log => <StatusBadge tone={statusToneForLog(log.level)}>{log.level}</StatusBadge> },
          { key: 'source', label: 'Source', width: '0.8fr' },
          { key: 'message', label: 'Message', width: '2fr' },
          { key: 'actions', label: 'Actions', width: '86px', render: log => <ActionCluster actions={['Inspect', 'Detail', 'More']} onAction={action => {
            actions.selectLog(log.key);
            openActionFocus(actions, action, `${log.source} ${log.time}`, log.message, log.level === 'WARN' ? 'yellow' : log.level === 'ERROR' ? 'red' : 'blue');
          }} /> }
        ]}
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Info', predicate: log => log.level === 'INFO' },
          { label: 'Warn', predicate: log => log.level === 'WARN' },
          { label: 'Success', predicate: log => log.level === 'SUCCESS' }
        ]}
        getSearchText={log => `${log.time} ${log.level} ${log.source} ${log.message}`}
        onSelect={log => actions.selectLog(log.key)}
        rowKey={log => log.key}
        rows={logRows}
        searchable
        searchPlaceholder="Search logs..."
        selectedRow={selectedLog}
      />
    </div>
  ), <LogInspector log={selectedLog} />);
}
