import {
  ActionCluster,
  DataTable,
  LevelMeter,
  MetricCard,
  Panel,
  StatusBadge
} from './AutomixerComponents';
import type { Channel, Metric, Tone, UiActions } from '../../types';

function channelMetrics(channels: Channel[]): Metric[] {
  const active = channels.filter(channel => channel.status === 'ACTIVE').length;
  const muted = channels.filter(channel => channel.status === 'MUTED').length;
  const inactive = channels.length - active - muted;
  const recognized = channels.filter(channel => channel.confidence > 0).length;
  return [
    { label: 'Total Channels', value: String(channels.length), sub: 'all channels', tone: 'blue' },
    { label: 'Active', value: String(active), sub: 'signal present', tone: 'green' },
    { label: 'Muted', value: String(muted), sub: 'operator mute', tone: 'yellow' },
    { label: 'Inactive', value: String(inactive), sub: 'no signal', tone: 'purple' },
    { label: 'Coverage', value: `${Math.round((recognized / Math.max(channels.length, 1)) * 100)}%`, sub: 'recognized', tone: 'blue' }
  ];
}

function toneForChannelAction(channel: Channel): Tone {
  if (channel.group === 'VOCALS') return 'purple';
  if (channel.group === 'BASS') return 'yellow';
  return 'blue';
}

function focusChannelAction(actions: UiActions, channel: Channel, action: string) {
  actions.selectChannel(channel.id);
  actions.focusAction({
    action,
    target: channel.name,
    detail: `${channel.source} · ${channel.group} · ${channel.level.toFixed(1)} dBFS. Channel action is local-only.`,
    tone: toneForChannelAction(channel),
    timestamp: '10:24:15'
  });
}

function statusTone(channel: Channel): Tone {
  if (channel.status === 'ACTIVE') return 'green';
  if (channel.status === 'MUTED') return 'yellow';
  return 'muted';
}

export function ChannelInspectorPanel({ channel }: { channel: Channel }) {
  return (
    <div className="flex h-full flex-col gap-2">
      <Panel>
        <div className="text-center">
          <div className="mx-auto mb-3 grid h-16 w-16 place-items-center rounded-full border border-ai bg-ai/10 text-xl font-black text-ai">MIC</div>
          <h3 className="text-[15px] font-bold">{channel.name}</h3>
          <p className="text-[10px] text-muted">{channel.source} · {channel.status}</p>
          <div className="mt-2"><StatusBadge tone="green">{channel.recognition}</StatusBadge></div>
        </div>
      </Panel>
      <Panel title="Overview">
        <div className="grid grid-cols-[78px_1fr] gap-y-2 text-[11px]">
          <span className="text-muted">Role</span><span>{channel.type}</span>
          <span className="text-muted">Group / Bus</span><span className="text-online">{channel.group}</span>
          <span className="text-muted">Fader</span><span>{channel.fader.toFixed(1)} dB</span>
          <span className="text-muted">Color</span><span className="text-ai">Purple</span>
          <span className="text-muted">Last Activity</span><span className="text-online">10:24:15</span>
        </div>
      </Panel>
      <Panel title="Level Meter" className="flex-1">
        <LevelMeter value={channel.level} />
        <div className="mt-4 space-y-3 text-[11px]">
          <div className="flex justify-between"><span className="text-muted">Input Level</span><b>{channel.level.toFixed(1)} dBFS</b></div>
          <div className="flex justify-between"><span className="text-muted">LUFS Short</span><b>{channel.lufs.toFixed(1)} LUFS</b></div>
          <div className="flex justify-between"><span className="text-muted">True Peak</span><b>-1.1 dBTP</b></div>
          <div className="flex justify-between"><span className="text-muted">Dynamic Range</span><b>{channel.dynamic.toFixed(1)} dB</b></div>
        </div>
      </Panel>
    </div>
  );
}

export function ChannelListWorkspace({
  channels,
  selectedChannel,
  actions
}: {
  channels: Channel[];
  selectedChannel: Channel;
  actions: UiActions;
}) {
  return (
    <div className="grid h-full grid-rows-[76px_1fr] gap-2">
      <div className="grid grid-cols-5 gap-2">{channelMetrics(channels).map(metric => <MetricCard metric={metric} key={metric.label} />)}</div>
      <DataTable
        rows={channels}
        selectedRow={selectedChannel}
        onSelect={channel => actions.selectChannel(channel.id)}
        rowKey={channel => channel.id}
        searchable
        searchPlaceholder="Search channel..."
        filters={[
          { label: 'All', predicate: () => true },
          { label: 'Active', predicate: channel => channel.status === 'ACTIVE' },
          { label: 'Muted', predicate: channel => channel.status === 'MUTED' },
          { label: 'Vocals', predicate: channel => channel.group === 'VOCALS' },
          { label: 'Drums', predicate: channel => channel.group === 'DRUMS' }
        ]}
        getSearchText={channel => `${channel.id} ${channel.name} ${channel.type} ${channel.source} ${channel.group} ${channel.recognition}`}
        columns={[
          { key: 'id', label: '#', width: '34px' },
          { key: 'name', label: 'Channel Name', width: '1.2fr' },
          { key: 'type', label: 'Type / Role', width: '0.9fr' },
          { key: 'source', label: 'Source', width: '0.7fr' },
          { key: 'level', label: 'Input Level', width: '1.1fr', render: channel => <LevelMeter value={channel.level} /> },
          { key: 'lufs', label: 'LUFS', width: '0.7fr' },
          { key: 'dynamic', label: 'Dynamic', width: '0.75fr' },
          { key: 'status', label: 'Status', width: '0.8fr', render: channel => <StatusBadge tone={statusTone(channel)}>{channel.status}</StatusBadge> },
          { key: 'recognition', label: 'AI Recognition', width: '1.2fr' },
          { key: 'group', label: 'Group / Bus', width: '0.8fr' },
          { key: 'actions', label: 'Actions', width: '86px', render: channel => <ActionCluster actions={['Inspect', 'Meter', 'Route']} onAction={action => focusChannelAction(actions, channel, action)} /> }
        ]}
      />
    </div>
  );
}
