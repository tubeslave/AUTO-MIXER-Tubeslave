#!/usr/bin/env python3
"""
Restore channel settings from backup file
"""
from wing_client import WingClient
import json
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def restore_channel(client: WingClient, channel_data: dict):
    """Restore a single channel's settings"""
    ch = channel_data['channel']
    
    logger.info(f"Restoring channel {ch}...")
    
    # Basic channel info
    if channel_data.get('name'):
        client.set_channel_name(ch, channel_data['name'])
    if channel_data.get('color'):
        client.set_channel_color(ch, channel_data['color'])
    
    # Input settings
    input_settings = channel_data.get('input', {})
    client.set_channel_gain(ch, input_settings.get('trim', 0.0))
    client.set_channel_balance(ch, input_settings.get('balance', 0.0))
    client.set_channel_phase_invert(ch, input_settings.get('phase_invert', 0))
    time.sleep(0.01)
    
    # Delay
    delay_mode = input_settings.get('delay_mode', 'MS')
    delay_value = input_settings.get('delay_value', 0.5)
    delay_on = input_settings.get('delay_on', 0)
    client.send(f"/ch/{ch}/in/set/dlymode", delay_mode)
    client.send(f"/ch/{ch}/in/set/dly", delay_value)
    client.send(f"/ch/{ch}/in/set/dlyon", delay_on)
    time.sleep(0.01)
    
    # Channel controls
    controls = channel_data.get('controls', {})
    client.set_channel_fader(ch, controls.get('fader', 0.0))
    client.set_channel_pan(ch, controls.get('pan', 0.0))
    client.set_channel_width(ch, controls.get('width', 100.0))
    client.set_channel_mute(ch, controls.get('mute', 0))
    time.sleep(0.01)
    
    # Filters
    filters = channel_data.get('filters', {})
    client.send(f"/ch/{ch}/flt/mdl", filters.get('model', 'TILT'))
    client.set_low_cut(
        ch,
        enabled=filters.get('low_cut', {}).get('enabled', 0),
        frequency=filters.get('low_cut', {}).get('frequency', 20.0),
        slope=str(filters.get('low_cut', {}).get('slope', 12))
    )
    client.set_high_cut(
        ch,
        enabled=filters.get('high_cut', {}).get('enabled', 0),
        frequency=filters.get('high_cut', {}).get('frequency', 20000.0),
        slope=str(filters.get('high_cut', {}).get('slope', 12))
    )
    client.send(f"/ch/{ch}/flt/tf", filters.get('tool_filter', 0))
    client.send(f"/ch/{ch}/flt/tilt", filters.get('tilt', 0.0))
    time.sleep(0.01)
    
    # EQ
    eq = channel_data.get('eq', {})
    client.set_eq_on(ch, eq.get('on', 0))
    if eq.get('on', 0) == 1:
        client.set_eq_model(ch, eq.get('model', 'STD'))
        client.set_eq_mix(ch, eq.get('mix', 100.0))
        
        # Low shelf
        ls = eq.get('low_shelf', {})
        client.set_eq_low_shelf(
            ch,
            gain=ls.get('gain', 0.0),
            freq=ls.get('freq', 100.0),
            q=ls.get('q', 1.0),
            eq_type=ls.get('type', 'PEQ')
        )
        
        # Bands 1-4
        for band_num in range(1, 5):
            band = eq.get('bands', {}).get(band_num, {})
            client.set_eq_band(
                ch,
                band_num,
                freq=band.get('freq', 1000.0),
                gain=band.get('gain', 0.0),
                q=band.get('q', 1.0)
            )
        
        # High shelf
        hs = eq.get('high_shelf', {})
        client.set_eq_high_shelf(
            ch,
            gain=hs.get('gain', 0.0),
            freq=hs.get('freq', 10000.0),
            q=hs.get('q', 1.0),
            eq_type=hs.get('type', 'SHV')
        )
    time.sleep(0.01)
    
    # Compressor
    comp = channel_data.get('compressor', {})
    client.set_compressor_on(ch, comp.get('on', 0))
    if comp.get('on', 0) == 1:
        client.set_compressor_model(ch, comp.get('model', 'COMP'))
        client.set_compressor(
            ch,
            threshold=comp.get('threshold', -10.0),
            ratio=str(comp.get('ratio', '3.0')),
            knee=comp.get('knee', 0),
            attack=comp.get('attack', 10.0),
            hold=comp.get('hold', 20.0),
            release=comp.get('release', 100.0),
            gain=comp.get('gain', 0.0),
            mix=comp.get('mix', 100.0),
            det=comp.get('detection', 'PEAK'),
            env=comp.get('envelope', 'LIN'),
            auto=comp.get('auto', 0)
        )
    time.sleep(0.01)
    
    # Gate
    gate = channel_data.get('gate', {})
    client.set_gate_on(ch, gate.get('on', 0))
    if gate.get('on', 0) == 1:
        client.set_gate_model(ch, gate.get('model', 'GATE'))
        client.set_gate(
            ch,
            threshold=gate.get('threshold', -40.0),
            range_db=gate.get('range', 10.0),
            attack=gate.get('attack', 5.0),
            hold=gate.get('hold', 10.0),
            release=gate.get('release', 100.0),
            accent=gate.get('accent', 50.0),
            ratio=str(gate.get('ratio', 'GATE'))
        )
    time.sleep(0.01)
    
    # Inserts
    inserts = channel_data.get('inserts', {})
    
    # Pre-insert
    if inserts.get('pre_insert'):
        pre_ins = inserts['pre_insert']
        fx_slot = pre_ins.get('slot')
        if fx_slot and fx_slot != 'NONE':
            client.send(f"/ch/{ch}/preins/ins", fx_slot)
            client.send(f"/ch/{ch}/preins/on", pre_ins.get('on', 0))
            if pre_ins.get('on', 0) == 1:
                fx_module = pre_ins.get('fx_module', {})
                if fx_module.get('model'):
                    client.set_fx_model(fx_slot, fx_module['model'])
                client.set_fx_on(fx_slot, fx_module.get('on', 0))
                client.set_fx_mix(fx_slot, fx_module.get('mix', 100.0))
                # Restore FX parameters
                fx_params = fx_module.get('parameters', {})
                for param_num, param_value in fx_params.items():
                    client.set_fx_parameter(fx_slot, param_num, param_value)
            time.sleep(0.05)
    
    # Post-insert
    if inserts.get('post_insert'):
        post_ins = inserts['post_insert']
        fx_slot = post_ins.get('slot')
        if fx_slot and fx_slot != 'NONE':
            client.send(f"/ch/{ch}/postins/ins", fx_slot)
            client.send(f"/ch/{ch}/postins/mode", post_ins.get('mode', 'FX'))
            client.send(f"/ch/{ch}/postins/on", post_ins.get('on', 0))
            client.send(f"/ch/{ch}/postins/w", post_ins.get('weight', 0.0))
            if post_ins.get('on', 0) == 1:
                fx_module = post_ins.get('fx_module', {})
                if fx_module.get('model'):
                    client.set_fx_model(fx_slot, fx_module['model'])
                client.set_fx_on(fx_slot, fx_module.get('on', 0))
                client.set_fx_mix(fx_slot, fx_module.get('mix', 100.0))
                # Restore FX parameters
                fx_params = fx_module.get('parameters', {})
                for param_num, param_value in fx_params.items():
                    client.set_fx_parameter(fx_slot, param_num, param_value)
            time.sleep(0.05)
    
    logger.info(f"Channel {ch} restored")


def restore_from_backup_using_client(client: "WingClient", backup_file: str, skip_confirm: bool = False) -> bool:
    """Restore all channels from backup file using an already-connected client (no prompt)."""
    try:
        with open(backup_file, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading backup file: {e}")
        return False
    logger.info(f"Loaded backup from: {backup_file}")
    channels = backup_data.get('channels', {})
    for ch_num, channel_data in channels.items():
        if 'error' in channel_data:
            continue
        try:
            restore_channel(client, channel_data)
        except Exception as e:
            logger.error(f"Error restoring channel {ch_num}: {e}")
    logger.info("Restore complete")
    return True


def restore_from_backup(backup_file: str, ip: str = "192.168.1.102", port: int = 2223):
    """Restore all channels from backup file"""
    
    # Load backup file
    try:
        with open(backup_file, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading backup file: {e}")
        return False
    
    logger.info(f"Loaded backup from: {backup_file}")
    logger.info(f"Backup timestamp: {backup_data.get('timestamp', 'Unknown')}")
    
    # Connect to Wing
    logger.info(f"Connecting to Wing at {ip}:{port}...")
    client = WingClient(ip, port)
    
    if not client.connect():
        logger.error("Failed to connect to Wing")
        return False
    
    logger.info("Connected! Waiting for initial scan...")
    time.sleep(5)
    
    # Confirm before proceeding
    print("\n" + "="*60)
    print("WARNING: This will restore ALL channel settings from backup!")
    print("="*60)
    confirm = input("\nType 'YES' to continue: ").strip()
    
    if confirm != 'YES':
        logger.info("Operation cancelled")
        client.disconnect()
        return False
    
    logger.info("\nStarting restoration...")
    
    # Restore all channels
    channels = backup_data.get('channels', {})
    for ch_num, channel_data in channels.items():
        if 'error' in channel_data:
            logger.warning(f"Skipping channel {ch_num}: {channel_data['error']}")
            continue
        try:
            restore_channel(client, channel_data)
            if int(ch_num) % 10 == 0:
                logger.info(f"Restored {ch_num}/40 channels...")
        except Exception as e:
            logger.error(f"Error restoring channel {ch_num}: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Restoration complete!")
    logger.info(f"{'='*60}\n")
    
    client.disconnect()
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: restore_channels.py <backup_file> [ip] [port]")
        sys.exit(1)
    
    backup_file = sys.argv[1]
    ip = sys.argv[2] if len(sys.argv) > 2 else "192.168.1.102"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 2223
    
    restore_from_backup(backup_file, ip, port)
