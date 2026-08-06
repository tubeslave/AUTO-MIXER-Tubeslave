"""
Audio Device Discovery Module

Lists available audio input devices and their channels.
Uses PyAudio for device enumeration.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def get_audio_devices() -> List[Dict[str, Any]]:
    """
    Get list of available audio input devices
    
    Returns:
        List of devices with their channels
    """
    devices = []
    pa = None
    
    try:
        import pyaudio
        
        pa = pyaudio.PyAudio()
        
        for idx in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(idx)
            
            # Only include input devices (maxInputChannels > 0)
            max_channels = int(device_info.get('maxInputChannels', 0))
            if max_channels > 0:
                channels = []
                for ch in range(1, max_channels + 1):
                    channels.append({
                        'id': ch,
                        'name': f'Input {ch}'
                    })
                
                devices.append({
                    'id': str(idx),
                    'name': device_info.get('name', f'Device {idx}'),
                    'channels': channels,
                    'sample_rate': int(device_info.get('defaultSampleRate', 48000)),
                    'max_channels': max_channels
                })
        
        logger.info(f"Found {len(devices)} audio input devices")
        
    except ImportError:
        logger.warning("PyAudio not installed; audio device discovery is unavailable")
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        devices = []
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception as exc:
                logger.warning("Failed to terminate PyAudio after discovery: %s", exc)
    
    return devices


def get_device_by_id(device_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific audio device by ID"""
    devices = get_audio_devices()
    for device in devices:
        if device['id'] == device_id:
            return device
    return None
