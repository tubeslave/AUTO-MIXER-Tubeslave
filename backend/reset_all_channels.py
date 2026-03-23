#!/usr/bin/env python3
"""
Test script: Reset all 40 channels to default settings
- Set trim = 0 dB
- Set fader = 0 dB
- Turn off all modules (EQ, Compressor, Gate)
- Reset all module parameters to defaults
- Turn off inserts
"""
from wing_client import WingClient
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def reset_channel(client: WingClient, channel: int):
    """Reset a single channel to default settings"""
    ch = channel
    
    logger.info(f"Resetting channel {ch}...")
    
    # 1. Set trim = 0 dB
    client.set_channel_gain(ch, 0.0)
    time.sleep(0.01)
    
    # 1a. Reset phase to normal (not inverted)
    client.set_channel_phase_invert(ch, 0)  # 0 = normal phase, 1 = inverted
    time.sleep(0.01)
    
    # 1b. Turn off delay and reset delay values
    # First set delay mode, then value, then turn off
    client.send(f"/ch/{ch}/in/set/dlymode", "MS")  # Set delay mode to MS (milliseconds)
    time.sleep(0.01)
    client.send(f"/ch/{ch}/in/set/dly", 0.5)  # Reset delay value to minimum (0.5ms for MS mode)
    time.sleep(0.01)
    client.send(f"/ch/{ch}/in/set/dlyon", 0)  # Turn off delay
    time.sleep(0.01)
    
    # 1c. Reset balance to 0
    client.set_channel_balance(ch, 0.0)
    time.sleep(0.01)
    
    # 2. Set fader = 0 dB
    client.set_channel_fader(ch, 0.0)
    time.sleep(0.01)
    
    # 3. Turn off EQ
    client.set_eq_on(ch, 0)
    time.sleep(0.01)
    
    # 4. Reset EQ parameters to defaults
    # Low shelf: gain=0, freq=default, q=default
    client.set_eq_low_shelf(ch, gain=0.0)
    time.sleep(0.01)
    
    # Bands 1-4: gain=0
    for band in range(1, 5):
        client.set_eq_band(ch, band, gain=0.0)
        time.sleep(0.01)
    
    # High shelf: gain=0
    client.set_eq_high_shelf(ch, gain=0.0)
    time.sleep(0.01)
    
    # 5. Turn off Compressor/Dynamics
    client.set_compressor_on(ch, 0)
    time.sleep(0.01)
    
    # 6. Reset compressor parameters to defaults
    client.set_compressor(
        ch,
        threshold=-10.0,  # Default threshold
        ratio="3.0",       # Default ratio
        attack=10.0,       # Default attack
        release=100.0,     # Default release
        gain=0.0,          # No make-up gain
        mix=100.0          # 100% mix
    )
    time.sleep(0.01)
    
    # 7. Turn off Gate
    client.set_gate_on(ch, 0)
    time.sleep(0.01)
    
    # 8. Reset gate parameters to defaults
    client.set_gate(
        ch,
        threshold=-40.0,   # Default threshold
        range_db=10.0,      # Default range
        attack=5.0,         # Default attack
        release=100.0,      # Default release
        accent=50.0         # Default accent
    )
    time.sleep(0.01)
    
    # 9. Turn off filters and reset filter model
    client.set_low_cut(ch, enabled=0)
    client.set_high_cut(ch, enabled=0)
    time.sleep(0.01)
    
    # Reset filter model to TILT (default) or NONE
    client.send(f"/ch/{ch}/flt/mdl", "TILT")
    time.sleep(0.01)
    
    # Turn off tool filter
    client.send(f"/ch/{ch}/flt/tf", 0)
    time.sleep(0.01)
    
    # Reset filter tilt to 0
    client.send(f"/ch/{ch}/flt/tilt", 0.0)
    time.sleep(0.01)
    
    # 10. Reset pan to center (0)
    client.set_channel_pan(ch, 0.0)
    time.sleep(0.01)
    
    # 11. Reset width to 100%
    client.set_channel_width(ch, 100.0)
    time.sleep(0.01)
    
    # 12. Turn off inserts and reset FX module parameters
    preins_slot = client.state.get(f'/ch/{ch}/preins/ins')
    postins_slot = client.state.get(f'/ch/{ch}/postins/ins')
    
    if preins_slot and preins_slot != 'NONE':
        # Turn off pre-insert
        client.send(f"/ch/{ch}/preins/on", 0)
        time.sleep(0.01)
        # Reset FX module parameters to defaults (set all to 0 or default values)
        reset_fx_module(client, preins_slot)
    
    if postins_slot and postins_slot != 'NONE':
        # Turn off post-insert
        client.send(f"/ch/{ch}/postins/on", 0)
        time.sleep(0.01)
        # Reset FX module parameters to defaults
        reset_fx_module(client, postins_slot)
    
    logger.info(f"Channel {ch} reset complete")


def reset_fx_module(client: WingClient, fx_slot: str):
    """Reset FX module parameters to default values"""
    fx_num = fx_slot.replace('FX', '') if fx_slot.startswith('FX') else fx_slot
    
    # Turn off FX module
    client.set_fx_on(fx_slot, 0)
    time.sleep(0.01)
    
    # Reset mix to 100%
    client.set_fx_mix(fx_slot, 100.0)
    time.sleep(0.01)
    
    # Reset all numbered parameters to 0 (default)
    # Most FX modules use parameters 1-32
    for param_num in range(1, 33):
        try:
            client.set_fx_parameter(fx_slot, param_num, 0.0)
            if param_num % 8 == 0:  # Small delay every 8 parameters
                time.sleep(0.01)
        except Exception as e:
            logger.debug(f"Could not reset FX{fx_num} parameter {param_num}: {e}")
    
    time.sleep(0.05)


def reset_all_channels(auto_confirm=False):
    """Reset all 40 channels"""
    import sys
    
    # Connect to Wing - allow command line args
    if len(sys.argv) >= 2:
        ip = sys.argv[1]
    else:
        ip = input("Enter Wing IP address [192.168.1.102]: ").strip() or "192.168.1.102"
    
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    else:
        port = int(input("Enter OSC port [2223]: ").strip() or "2223")
    
    logger.info(f"Connecting to Wing at {ip}:{port}...")
    client = WingClient(ip, port)
    
    if not client.connect():
        logger.error("Failed to connect to Wing")
        return
    
    logger.info("Connected! Waiting for initial scan...")
    time.sleep(5)  # Wait for initial scan and channel scan
    
    # Confirm before proceeding (unless auto_confirm)
    if not auto_confirm:
        print("\n" + "="*60)
        print("WARNING: This will reset ALL settings on ALL 40 channels!")
        print("  - Trim = 0 dB")
        print("  - Fader = 0 dB")
        print("  - All modules OFF (EQ, Compressor, Gate)")
        print("  - All module parameters reset to defaults")
        print("  - All inserts OFF")
        print("  - FX module parameters reset")
        print("="*60)
        confirm = input("\nType 'YES' to continue: ").strip()
        
        if confirm != 'YES':
            logger.info("Operation cancelled")
            client.disconnect()
            return
    
    logger.info("\nStarting reset of all 40 channels...")
    logger.info("This may take a minute...\n")
    
    start_time = time.time()
    
    # Reset all channels
    for ch in range(1, 41):
        try:
            reset_channel(client, ch)
            if ch % 10 == 0:
                logger.info(f"Progress: {ch}/40 channels reset")
        except Exception as e:
            logger.error(f"Error resetting channel {ch}: {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Reset complete! Processed 40 channels in {elapsed:.1f} seconds")
    logger.info(f"{'='*60}\n")
    
    client.disconnect()
    logger.info("Disconnected")


if __name__ == "__main__":
    import sys
    # Allow auto-confirm with --yes flag
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    reset_all_channels(auto_confirm=auto_confirm)
