#!/usr/bin/env python3
"""
Example: How to read channel inserts and send correction parameters to FX modules

This demonstrates:
1. Reading all channels with inserts
2. Getting FX module information
3. Sending correction parameters to FX modules
"""
from wing_client import WingClient
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_read_and_correct():
    """Example: Read channel inserts and send corrections"""
    
    # Connect to Wing
    client = WingClient('192.168.1.102', 2223)
    
    if not client.connect():
        logger.error("Failed to connect to Wing")
        return
    
    logger.info("Connected! Waiting for channel scan...")
    time.sleep(8)  # Wait for background scan to complete
    
    # Get all channels with inserts
    channels_with_inserts = client.get_all_channels_with_inserts()
    
    logger.info(f"\nFound {len(channels_with_inserts)} channels with inserts:")
    
    for ch_num, ch_info in channels_with_inserts.items():
        logger.info(f"\n--- Channel {ch_num} ---")
        
        # Check pre-insert
        if ch_info['inserts']['pre_insert']:
            fx = ch_info['inserts']['pre_insert']['fx_module']
            logger.info(f"Pre Insert: {fx['slot']} - Model: {fx.get('model', 'N/A')}")
            logger.info(f"  Current parameters: {fx.get('parameters', {})}")
            
            # Example: Adjust FX13 (P-BASS) parameter 1 (intensity)
            if fx['slot'] == 'FX13' and fx.get('model') == 'P-BASS':
                current_intensity = fx.get('parameters', {}).get(1)
                if current_intensity is not None:
                    logger.info(f"  Current intensity: {current_intensity} dB")
                    # Example correction: increase intensity by 1 dB
                    # new_intensity = current_intensity + 1.0
                    # client.set_fx_parameter('FX13', 1, new_intensity)
                    # logger.info(f"  Set intensity to: {new_intensity} dB")
        
        # Check post-insert
        if ch_info['inserts']['post_insert']:
            fx = ch_info['inserts']['post_insert']['fx_module']
            logger.info(f"Post Insert: {fx['slot']} - Model: {fx.get('model', 'N/A')}")
            logger.info(f"  Current parameters: {fx.get('parameters', {})}")
    
    # Example: Get specific channel insert info
    logger.info("\n--- Channel 10 Detailed Info ---")
    ch10_inserts = client.get_channel_inserts(10)
    if ch10_inserts['pre_insert']:
        fx = ch10_inserts['pre_insert']['fx_module']
        logger.info(f"FX Module: {fx['slot']}")
        logger.info(f"  Model: {fx.get('model')}")
        logger.info(f"  On: {fx.get('on')}")
        logger.info(f"  Mix: {fx.get('mix')}%")
        logger.info(f"  Parameters: {fx.get('parameters', {})}")
        
        # Example: Read specific parameter
        param1 = client.get_fx_parameter('FX13', 1)
        logger.info(f"  Parameter 1 (intensity): {param1} dB")
        
        # Example: Send correction (commented out to avoid changing settings)
        # client.set_fx_parameter('FX13', 1, -1.0)  # Set intensity to -1.0 dB
        # logger.info("  Sent correction: Parameter 1 = -1.0 dB")
    
    client.disconnect()
    logger.info("\nDisconnected")


if __name__ == "__main__":
    example_read_and_correct()
