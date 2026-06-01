import { DataTable, LevelMeter, MetricCard, Panel, Sparkline, StatusBadge } from '../components/prototype/AutomixerComponents';
import { ChannelInspectorPanel, ChannelListWorkspace } from '../components/prototype/ChannelListPanels';
import { buses, channels } from '../mock/automixerData';
import type { OperatorViewData } from '../services/useOperatorBackend';
import type { BusRecord, Metric, UiActions, UiState } from '../types';
import { formatDb, formatLufs, readOnlyEmptyState, rowsForReadModelSource, sourceDisplay, statusToneForBus, tableView, TechnicalChecklist } from './viewHelpers';

export function ChannelList(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const selectedChannel = data.channels.find(channel => channel.id === ui.selectedChannelId) ?? data.channels[0] ?? channels[11];
  return tableView('8. Channel List', 'All input channels and current status', (
    <ChannelListWorkspace actions={actions} channels={data.channels} selectedChannel={selectedChannel} />
  ), <ChannelInspectorPanel channel={selectedChannel} />);
}

function GroupsRightPanel({ bus, sourceLabel, sourceRaw }: { bus?: BusRecord; sourceLabel: string; sourceRaw: string }) {
  const members = bus?.members ?? [];
  return (
    <div className="flex flex-col gap-2">
      <Panel title="Bus Details">
        {bus ? (
          <>
            <div className="mb-3 flex items-center gap-3">
              <div className={`grid h-12 w-12 place-items-center rounded-full border bg-panel2 text-[13px] font-black ${bus.color === 'purple' ? 'border-ai text-ai' : bus.color === 'blue' ? 'border-audio text-audio' : bus.color === 'yellow' ? 'border-warn text-warn' : 'border-online text-online'}`}>
                {bus.name.slice(0, 2).toUpperCase()}
              </div>
              <div className="min-w-0">
                <div className="truncate text-[14px] font-bold text-text">{bus.name}</div>
                <div className="text-[10px] text-muted">{bus.role}</div>
                <div className="mt-1"><StatusBadge tone={statusToneForBus(bus.status)}>{bus.status ?? 'Group'}</StatusBadge></div>
              </div>
            </div>
            <TechnicalChecklist rows={[
              { label: 'Source', value: sourceLabel, tone: sourceLabel === 'Backend DTO' ? 'green' : 'blue' },
              { label: 'Channels', value: String(bus.channels), tone: 'blue' },
              { label: 'Active', value: String(bus.activeChannels ?? bus.channels), tone: 'green' },
              { label: 'Muted', value: String(bus.mutedChannels ?? 0), tone: (bus.mutedChannels ?? 0) ? 'yellow' : 'green' },
              { label: 'Confidence', value: `${bus.confidence ?? 0}%`, tone: (bus.confidence ?? 0) > 85 ? 'green' : 'yellow' }
            ]} />
          </>
        ) : (
          <div>
            <h3 className="mb-2 text-[14px] font-bold text-text">No bus selected</h3>
            {readOnlyEmptyState('No group/bus rows received', sourceRaw)}
          </div>
        )}
      </Panel>
      <Panel title="Members">
        <div className="space-y-2">
          {(members.length ? members : ['No channel inventory rows']).slice(0, 10).map((member, index) => (
            <div className="grid grid-cols-[28px_1fr] gap-2 border-b border-line pb-2 text-[10px]" key={`${member}-${index}`}>
              <span className="font-mono text-audio">{String(index + 1).padStart(2, '0')}</span>
              <span className="truncate text-muted">{member}</span>
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="Write Guard">
        <TechnicalChecklist rows={[
          { label: 'Bus writes', value: 'Read-only', tone: 'blue' },
          { label: 'Routing changes', value: 'Blocked', tone: 'red' },
          { label: 'OSC direct path', value: 'Unavailable', tone: 'red' }
        ]} />
      </Panel>
    </div>
  );
}

export function Groups(ui: UiState, actions: UiActions, data: OperatorViewData) {
  const sourceRaw = data.readModelSources.groups;
  const busRows = rowsForReadModelSource(data.buses, buses, sourceRaw);
  const selectedBus = busRows.find(bus => bus.name === ui.selectedBusName) ?? busRows.find(bus => bus.status === 'Active') ?? busRows[0];
  const sourceLabel = sourceDisplay(sourceRaw);
  const activeChannels = busRows.reduce((sum, bus) => sum + (bus.activeChannels ?? bus.channels), 0);
  const mutedChannels = busRows.reduce((sum, bus) => sum + (bus.mutedChannels ?? 0), 0);
  const avgConfidence = Math.round(busRows.reduce((sum, bus) => sum + (bus.confidence ?? 0), 0) / Math.max(busRows.length, 1));
  const metrics: Metric[] = [
    { label: 'Buses', value: String(busRows.length), sub: 'from channel inventory', tone: 'blue' },
    { label: 'Active Channels', value: String(activeChannels), sub: 'assigned to buses', tone: 'green' },
    { label: 'Muted', value: String(mutedChannels), sub: 'held in group map', tone: mutedChannels ? 'yellow' : 'green' },
    { label: 'Avg Confidence', value: `${avgConfidence}%`, sub: 'recognition per bus', tone: avgConfidence > 85 ? 'green' : 'yellow' },
    { label: 'Source', value: sourceLabel, sub: data.readModelSources.groups, tone: sourceLabel === 'Backend DTO' ? 'green' : 'blue' }
  ];

  return tableView('9. Groups / Buses', 'Stem buses, routing roles, and group balance', (
    <div className="grid h-full grid-rows-[76px_1fr_118px] gap-2">
      <div className="grid grid-cols-5 gap-2">{metrics.map(metric => <MetricCard metric={metric} key={metric.label} />)}</div>
      <div className="grid min-h-0 grid-cols-[1.18fr_0.82fr] gap-2">
        <DataTable
          rows={busRows}
          selectedRow={selectedBus}
          onSelect={bus => actions.selectBus(bus.name)}
          searchable
          searchPlaceholder="Search bus or member..."
          filters={[
            { label: 'All', predicate: () => true },
            { label: 'Active', predicate: row => row.status === 'Active' },
            { label: 'Muted', predicate: row => row.status === 'Muted' },
            { label: 'Returns', predicate: row => row.role === 'Return' }
          ]}
          getSearchText={row => `${row.name} ${row.role} ${row.status} ${(row.members ?? []).join(' ')}`}
          rowKey={row => row.name}
          emptyState={readOnlyEmptyState('No group/bus rows received', sourceRaw)}
          columns={[
            { key: 'name', label: 'Bus', width: '1.1fr', render: row => <span className="font-semibold text-text">{row.name}</span> },
            { key: 'role', label: 'Role', width: '0.65fr' },
            { key: 'level', label: 'Level', width: '0.7fr', render: row => formatDb(row.level) },
            { key: 'lufs', label: 'LUFS', width: '0.78fr', render: row => formatLufs(row.lufs) },
            { key: 'channels', label: 'Ch', width: '0.45fr' },
            { key: 'status', label: 'Status', width: '0.75fr', render: row => <StatusBadge tone={statusToneForBus(row.status)}>{row.status ?? 'Group'}</StatusBadge> }
          ]}
        />
        <Panel title="Bus Balance">
          {busRows.length ? <div className="space-y-3">
            {busRows.map(bus => (
              <div className="grid grid-cols-[112px_1fr_70px] items-center gap-2 text-[11px]" key={bus.name}>
                <span className="truncate text-text">{bus.name}</span>
                <LevelMeter value={bus.level} />
                <span className="text-right font-mono text-muted">{formatLufs(bus.lufs)}</span>
              </div>
            ))}
          </div> : readOnlyEmptyState('No bus balance rows', sourceRaw)}
        </Panel>
      </div>
      <div className="grid grid-cols-4 gap-2">
        <Panel title="Bus Dynamics">{busRows.length ? <Sparkline values={busRows.map(bus => Math.max(8, Math.min(96, 72 + bus.level)))} tone="green" filled /> : readOnlyEmptyState('No bus dynamics rows', sourceRaw)}</Panel>
        <Panel title="Channel Assignment">{busRows.length ? <TechnicalChecklist rows={busRows.slice(0, 4).map(bus => ({ label: bus.name, value: `${bus.channels} ch`, tone: bus.color }))} /> : readOnlyEmptyState('No channel assignment rows', sourceRaw)}</Panel>
        <Panel title="Safety"><TechnicalChecklist rows={[{ label: 'Bus trims', value: 'Display only', tone: 'blue' }, { label: 'Routing', value: 'Read-only', tone: 'blue' }]} /></Panel>
        <Panel title="Source"><TechnicalChecklist rows={[{ label: 'Inventory', value: sourceLabel, tone: sourceLabel === 'Backend DTO' ? 'green' : 'blue' }, { label: 'Writes', value: 'Disabled', tone: 'red' }]} /></Panel>
      </div>
    </div>
  ), <GroupsRightPanel bus={selectedBus} sourceLabel={sourceLabel} sourceRaw={sourceRaw} />);
}
