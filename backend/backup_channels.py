#!/usr/bin/env python3
"""
Backup all channel settings including inserts and FX modules
Saves to JSON file for later restoration
"""
from wing_client import WingClient
import time
import json
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backup_channel(client: WingClient, channel: int) -> dict:
    """Backup a single channel's settings"""
    ch = channel
    channel_data = {
        'channel': ch,
        'timestamp': datetime.now().isoformat()
    }
    
    # Basic channel info
    channel_data['name'] = client.state.get(f'/ch/{ch}/name', '')
    channel_data['color'] = client.state.get(f'/ch/{ch}/col', 0)
    channel_data['icon'] = client.state.get(f'/ch/{ch}/icon', 0)
    
    # Input settings
    channel_data['input'] = {
        'trim': client.state.get(f'/ch/{ch}/in/set/trim', 0.0),
        'balance': client.state.get(f'/ch/{ch}/in/set/bal', 0.0),
        'phase_invert': client.state.get(f'/ch/{ch}/in/set/inv', 0),
        'delay_mode': client.state.get(f'/ch/{ch}/in/set/dlymode', 'MS'),
        'delay_value': client.state.get(f'/ch/{ch}/in/set/dly', 0.5),
        'delay_on': client.state.get(f'/ch/{ch}/in/set/dlyon', 0),
    }
    
    # Channel controls
    channel_data['controls'] = {
        'fader': client.state.get(f'/ch/{ch}/fdr', 0.0),
        'pan': client.state.get(f'/ch/{ch}/pan', 0.0),
        'width': client.state.get(f'/ch/{ch}/wid', 100.0),
        'mute': client.state.get(f'/ch/{ch}/mute', 0),
    }
    
    # Filters
    channel_data['filters'] = {
        'model': client.state.get(f'/ch/{ch}/flt/mdl', 'TILT'),
        'low_cut': {
            'enabled': client.state.get(f'/ch/{ch}/flt/lc', 0),
            'frequency': client.state.get(f'/ch/{ch}/flt/lcf', 20.0),
            'slope': client.state.get(f'/ch/{ch}/flt/lcs', 12),
        },
        'high_cut': {
            'enabled': client.state.get(f'/ch/{ch}/flt/hc', 0),
            'frequency': client.state.get(f'/ch/{ch}/flt/hcf', 20000.0),
            'slope': client.state.get(f'/ch/{ch}/flt/hcs', 12),
        },
        'tool_filter': client.state.get(f'/ch/{ch}/flt/tf', 0),
        'tilt': client.state.get(f'/ch/{ch}/flt/tilt', 0.0),
    }
    
    # EQ
    channel_data['eq'] = {
        'on': client.state.get(f'/ch/{ch}/eq/on', 0),
        'model': client.state.get(f'/ch/{ch}/eq/mdl', 'STD'),
        'mix': client.state.get(f'/ch/{ch}/eq/mix', 100.0),
        'low_shelf': {
            'gain': client.state.get(f'/ch/{ch}/eq/lg', 0.0),
            'freq': client.state.get(f'/ch/{ch}/eq/lf', 100.0),
            'q': client.state.get(f'/ch/{ch}/eq/lq', 1.0),
            'type': client.state.get(f'/ch/{ch}/eq/leq', 'PEQ'),
        },
        'bands': {}
    }
    
    # EQ Bands 1-4
    for band in range(1, 5):
        channel_data['eq']['bands'][band] = {
            'gain': client.state.get(f'/ch/{ch}/eq/{band}g', 0.0),
            'freq': client.state.get(f'/ch/{ch}/eq/{band}f', 1000.0),
            'q': client.state.get(f'/ch/{ch}/eq/{band}q', 1.0),
        }
    
    # EQ High shelf
    channel_data['eq']['high_shelf'] = {
        'gain': client.state.get(f'/ch/{ch}/eq/hg', 0.0),
        'freq': client.state.get(f'/ch/{ch}/eq/hf', 10000.0),
        'q': client.state.get(f'/ch/{ch}/eq/hq', 1.0),
        'type': client.state.get(f'/ch/{ch}/eq/heq', 'SHV'),
    }
    
    # Compressor/Dynamics
    channel_data['compressor'] = {
        'on': client.state.get(f'/ch/{ch}/dyn/on', 0),
        'model': client.state.get(f'/ch/{ch}/dyn/mdl', 'COMP'),
        'threshold': client.state.get(f'/ch/{ch}/dyn/thr', -10.0),
        'ratio': client.state.get(f'/ch/{ch}/dyn/ratio', '3.0'),
        'knee': client.state.get(f'/ch/{ch}/dyn/knee', 0),
        'attack': client.state.get(f'/ch/{ch}/dyn/att', 10.0),
        'hold': client.state.get(f'/ch/{ch}/dyn/hld', 20.0),
        'release': client.state.get(f'/ch/{ch}/dyn/rel', 100.0),
        'gain': client.state.get(f'/ch/{ch}/dyn/gain', 0.0),
        'mix': client.state.get(f'/ch/{ch}/dyn/mix', 100.0),
        'detection': client.state.get(f'/ch/{ch}/dyn/det', 'PEAK'),
        'envelope': client.state.get(f'/ch/{ch}/dyn/env', 'LIN'),
        'auto': client.state.get(f'/ch/{ch}/dyn/auto', 0),
    }
    
    # Gate
    channel_data['gate'] = {
        'on': client.state.get(f'/ch/{ch}/gate/on', 0),
        'model': client.state.get(f'/ch/{ch}/gate/mdl', 'GATE'),
        'threshold': client.state.get(f'/ch/{ch}/gate/thr', -40.0),
        'range': client.state.get(f'/ch/{ch}/gate/range', 10.0),
        'attack': client.state.get(f'/ch/{ch}/gate/att', 5.0),
        'hold': client.state.get(f'/ch/{ch}/gate/hld', 10.0),
        'release': client.state.get(f'/ch/{ch}/gate/rel', 100.0),
        'accent': client.state.get(f'/ch/{ch}/gate/acc', 50.0),
        'ratio': client.state.get(f'/ch/{ch}/gate/ratio', 'GATE'),
    }
    
    # Inserts
    channel_data['inserts'] = client.get_channel_inserts(ch)
    
    return channel_data


def backup_all_channels(ip: str = "192.168.1.102", port: int = 2223, output_file: str = None):
    """Backup all 40 channels"""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../presets/channel_backup_{timestamp}.json"
    
    # Ensure presets directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    logger.info(f"Connecting to Wing at {ip}:{port}...")
    client = WingClient(ip, port)
    
    if not client.connect():
        logger.error("Failed to connect to Wing")
        return None
    
    logger.info("Connected! Waiting for channel scan...")
    time.sleep(8)  # Wait for full channel scan
    
    logger.info("Starting backup of all 40 channels...")
    
    backup_data = {
        'timestamp': datetime.now().isoformat(),
        'wing_ip': ip,
        'wing_port': port,
        'channels': {}
    }
    
    # Backup all channels
    for ch in range(1, 41):
        try:
            channel_data = backup_channel(client, ch)
            backup_data['channels'][ch] = channel_data
            if ch % 10 == 0:
                logger.info(f"Backed up {ch}/40 channels...")
        except Exception as e:
            logger.error(f"Error backing up channel {ch}: {e}")
            backup_data['channels'][ch] = {'error': str(e)}
    
    # Save to file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(output_file)
        logger.info(f"\n{'='*60}")
        logger.info(f"Backup complete!")
        logger.info(f"File: {output_file}")
        logger.info(f"Size: {file_size / 1024:.1f} KB")
        logger.info(f"Channels backed up: {len(backup_data['channels'])}")
        logger.info(f"{'='*60}\n")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error saving backup file: {e}")
        return None
    
    finally:
        client.disconnect()


if __name__ == "__main__":
    import sys
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 2223
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    backup_all_channels(ip, port, output_file)
