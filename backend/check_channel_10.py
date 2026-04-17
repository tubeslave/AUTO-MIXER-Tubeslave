#!/usr/bin/env python3
"""
Query and display all settings for channel 10 on the Wing mixer
"""
from wing_client import WingClient
from wing_addresses import WING_OSC_ADDRESSES
import time
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_fx_addresses(fx_slot: str):
    """Get FX module addresses for a given FX slot (e.g., 'FX1', 'FX2', etc.)"""
    addresses = []
    fx_num = fx_slot.replace('FX', '') if fx_slot.startswith('FX') else fx_slot
    
    # Basic FX info - according to WING documentation, model is at /fx/{n}/mdl
    addresses.extend([
        f"/fx/{fx_num}/mdl",      # FX model (P-BASS, REVERB, etc.)
        f"/fx/{fx_num}/on",       # FX on/off
        f"/fx/{fx_num}/fxmix",    # FX mix
        f"/fx/{fx_num}/name",     # FX name (if available)
        f"/fx/{fx_num}/type",     # FX type (if available)
    ])
    
    # Common FX parameters (these vary by FX type, but we'll query common ones)
    # Most FX have these basic parameters
    common_params = [
        'rate', 'depth', 'feedback', 'time', 'size', 'predelay', 'damp', 
        'low', 'high', 'freq', 'gain', 'q', 'threshold', 'ratio', 
        'attack', 'release', 'speed', 'width', 'tone', 'drive', 'wet', 'dry',
        'decay', 'room', 'early', 'late', 'diffusion', 'mod', 'chorus',
        'delay', 'reverb', 'filter', 'cutoff', 'resonance'
    ]
    for param in common_params:
        addresses.append(f"/fx/{fx_num}/{param}")
    
    # Query numbered parameters (WING uses numbered parameters for FX)
    for i in range(1, 33):  # Query parameters 1-32 (most FX have up to 32 params)
        addresses.append(f"/fx/{fx_num}/{i}")
    
    # Also try node-based querying
    addresses.append(f"/fx/{fx_num}/node")
    
    return addresses


def get_all_channel_addresses(channel: int):
    """Get all OSC addresses for a channel"""
    addresses = []
    ch = channel
    
    # Basic channel info
    addresses.extend([
        f"/ch/{ch}/name",
        f"/ch/{ch}/icon",
        f"/ch/{ch}/col",
        f"/ch/{ch}/led",
        f"/ch/{ch}/tags",
    ])
    
    # Input settings
    addresses.extend([
        f"/ch/{ch}/in/set/mode",
        f"/ch/{ch}/in/set/srcauto",
        f"/ch/{ch}/in/set/altsrc",
        f"/ch/{ch}/in/set/inv",
        f"/ch/{ch}/in/set/trim",
        f"/ch/{ch}/in/set/bal",
        f"/ch/{ch}/in/set/g",
        f"/ch/{ch}/in/set/vph",
        f"/ch/{ch}/in/set/dlymode",
        f"/ch/{ch}/in/set/dly",
        f"/ch/{ch}/in/set/dlyon",
    ])
    
    # Input connection
    addresses.extend([
        f"/ch/{ch}/in/conn/grp",
        f"/ch/{ch}/in/conn/in",
        f"/ch/{ch}/in/conn/altgrp",
        f"/ch/{ch}/in/conn/altin",
    ])
    
    # Filters
    addresses.extend([
        f"/ch/{ch}/flt/lc",
        f"/ch/{ch}/flt/lcf",
        f"/ch/{ch}/flt/lcs",
        f"/ch/{ch}/flt/hc",
        f"/ch/{ch}/flt/hcf",
        f"/ch/{ch}/flt/hcs",
        f"/ch/{ch}/flt/tf",
        f"/ch/{ch}/flt/mdl",
        f"/ch/{ch}/flt/tilt",
    ])
    
    # Channel controls
    addresses.extend([
        f"/ch/{ch}/mute",
        f"/ch/{ch}/fdr",
        f"/ch/{ch}/pan",
        f"/ch/{ch}/wid",
        f"/ch/{ch}/solo",
        f"/ch/{ch}/sololed",
        f"/ch/{ch}/solosafe",
        f"/ch/{ch}/mon",
        f"/ch/{ch}/proc",
        f"/ch/{ch}/ptap",
        f"/ch/{ch}/presolo",
    ])
    
    # PreSend EQ
    addresses.extend([
        f"/ch/{ch}/peq/on",
        f"/ch/{ch}/peq/1g",
        f"/ch/{ch}/peq/1f",
        f"/ch/{ch}/peq/1q",
        f"/ch/{ch}/peq/2g",
        f"/ch/{ch}/peq/2f",
        f"/ch/{ch}/peq/2q",
        f"/ch/{ch}/peq/3g",
        f"/ch/{ch}/peq/3f",
        f"/ch/{ch}/peq/3q",
    ])
    
    # Gate
    addresses.extend([
        f"/ch/{ch}/gate/on",
        f"/ch/{ch}/gate/mdl",
        f"/ch/{ch}/gate/thr",
        f"/ch/{ch}/gate/range",
        f"/ch/{ch}/gate/att",
        f"/ch/{ch}/gate/hld",
        f"/ch/{ch}/gate/rel",
        f"/ch/{ch}/gate/acc",
        f"/ch/{ch}/gate/ratio",
    ])
    
    # EQ
    addresses.extend([
        f"/ch/{ch}/eq/on",
        f"/ch/{ch}/eq/mdl",
        f"/ch/{ch}/eq/mix",
        f"/ch/{ch}/eq/lg",
        f"/ch/{ch}/eq/lf",
        f"/ch/{ch}/eq/lq",
        f"/ch/{ch}/eq/leq",
        f"/ch/{ch}/eq/1g",
        f"/ch/{ch}/eq/1f",
        f"/ch/{ch}/eq/1q",
        f"/ch/{ch}/eq/2g",
        f"/ch/{ch}/eq/2f",
        f"/ch/{ch}/eq/2q",
        f"/ch/{ch}/eq/3g",
        f"/ch/{ch}/eq/3f",
        f"/ch/{ch}/eq/3q",
        f"/ch/{ch}/eq/4g",
        f"/ch/{ch}/eq/4f",
        f"/ch/{ch}/eq/4q",
        f"/ch/{ch}/eq/hg",
        f"/ch/{ch}/eq/hf",
        f"/ch/{ch}/eq/hq",
        f"/ch/{ch}/eq/heq",
    ])
    
    # Dynamics (Compressor)
    addresses.extend([
        f"/ch/{ch}/dyn/on",
        f"/ch/{ch}/dyn/mdl",
        f"/ch/{ch}/dyn/mix",
        f"/ch/{ch}/dyn/gain",
        f"/ch/{ch}/dyn/thr",
        f"/ch/{ch}/dyn/ratio",
        f"/ch/{ch}/dyn/knee",
        f"/ch/{ch}/dyn/det",
        f"/ch/{ch}/dyn/att",
        f"/ch/{ch}/dyn/hld",
        f"/ch/{ch}/dyn/rel",
        f"/ch/{ch}/dyn/env",
        f"/ch/{ch}/dyn/auto",
    ])
    
    # Inserts
    addresses.extend([
        f"/ch/{ch}/preins/on",
        f"/ch/{ch}/preins/ins",
        f"/ch/{ch}/preins/$stat",
        f"/ch/{ch}/postins/on",
        f"/ch/{ch}/postins/mode",
        f"/ch/{ch}/postins/ins",
        f"/ch/{ch}/postins/w",
        f"/ch/{ch}/postins/$stat",
    ])
    
    return addresses


def format_value(address: str, value):
    """Format a value for display"""
    if value is None:
        return "N/A"
    
    if isinstance(value, (list, tuple)):
        if len(value) >= 3:
            # Wing returns [display_string, normalized_value, actual_value]
            return f"{value[0]} ({value[2]})"
        return str(value)
    
    return str(value)


# FX parameter definitions from WING documentation
FX_PARAM_DEFINITIONS = {
    'P-BASS': {
        1: ('int', 'Intensity', 'dB', '[-24, 6]'),
        2: ('bass', 'Bass Gain', 'dB', '[-60, 0]'),
        3: ('xf', 'Crossover Frequency', 'Hz', '[32, 200]'),
        4: ('solo', 'Solo', '', '[0, 1]'),
    },
    # Add more FX types as needed
}

def get_fx_param_name(fx_model: str, param_num: int) -> tuple:
    """Get parameter name and description for an FX model"""
    if fx_model in FX_PARAM_DEFINITIONS:
        if param_num in FX_PARAM_DEFINITIONS[fx_model]:
            return FX_PARAM_DEFINITIONS[fx_model][param_num]
    return (f'param{param_num}', f'Parameter {param_num}', '', '')


def print_fx_modules(fx_slots: list, state: dict):
    """Print FX module settings"""
    print(f"\n{'='*60}")
    print("FX MODULE SETTINGS")
    print(f"{'='*60}\n")
    
    for fx_slot in fx_slots:
        fx_num = fx_slot.replace('FX', '') if fx_slot.startswith('FX') else fx_slot
        
        print(f"🎛️  {fx_slot}")
        print("-" * 60)
        
        # Get FX info - model is at /fx/{n}/mdl according to WING documentation
        fx_model = state.get(f'/fx/{fx_num}/mdl')  # FX model (P-BASS, REVERB, etc.)
        fx_name = state.get(f'/fx/{fx_num}/name') or state.get(f'/fx/{fx_num}/$name')
        fx_type = state.get(f'/fx/{fx_num}/type') or state.get(f'/fx/{fx_num}/$type')
        fx_node = state.get(f'/fx/{fx_num}/')  # Node type indicator
        fx_on = state.get(f'/fx/{fx_num}/on')
        fx_mix = state.get(f'/fx/{fx_num}/fxmix')
        
        fx_on_str = "ON" if fx_on == 1 else "OFF" if fx_on == 0 else "Unknown"
        
        print(f"  Slot:        {fx_slot}")
        if fx_model:
            print(f"  Model:       {fx_model}")
        if fx_node:
            print(f"  Node Type:   {fx_node}")
        if fx_name and fx_name != 'N/A' and fx_name:
            print(f"  Name:        {fx_name}")
        if fx_type and fx_type != 'N/A' and fx_type:
            print(f"  Type:        {fx_type}")
        if fx_on is not None:
            print(f"  Status:      {fx_on_str}")
        if fx_mix is not None:
            print(f"  Mix:         {fx_mix} %")
        print()
        
        # Get all FX parameters
        fx_params = {}
        excluded_keys = [
            f'/fx/{fx_num}/name', f'/fx/{fx_num}/$name',
            f'/fx/{fx_num}/type', f'/fx/{fx_num}/$type',
            f'/fx/{fx_num}/on', f'/fx/{fx_num}/fxmix',
            f'/fx/{fx_num}/mdl',  # Model is shown separately
            f'/fx/{fx_num}/',  # Root node (contains node type)
            f'/fx/{fx_num}/node'
        ]
        for key, value in state.items():
            if key.startswith(f'/fx/{fx_num}/') and key not in excluded_keys:
                param_name = key.replace(f'/fx/{fx_num}/', '')
                # Skip empty param names (root node)
                if param_name:
                    fx_params[param_name] = value
        
        if fx_params:
            print("  Parameters:")
            # Sort parameters - put named params first, then numbered
            named_params = {k: v for k, v in fx_params.items() if not k.isdigit()}
            numbered_params = {k: v for k, v in fx_params.items() if k.isdigit()}
            
            for param, value in sorted(named_params.items()):
                print(f"    {param:20s} = {format_value(f'/fx/{fx_num}/{param}', value)}")
            
            if numbered_params:
                print("\n  Parameter Settings:")
                for param in sorted(numbered_params.keys(), key=lambda x: int(x)):
                    value = numbered_params[param]
                    param_num = int(param)
                    param_id, param_desc, unit, range_info = get_fx_param_name(fx_model or 'UNKNOWN', param_num)
                    unit_str = f" {unit}" if unit else ""
                    range_str = f" (range: {range_info})" if range_info else ""
                    solo_str = " (ON)" if param_num == 4 and value == 1 else " (OFF)" if param_num == 4 else ""
                    print(f"    {param_desc:25s} ({param_id}) = {value}{unit_str}{range_str}{solo_str}")
        else:
            print("  No parameters found")
        print()


def print_channel_settings(channel: int, state: dict):
    """Print formatted channel settings"""
    ch = channel
    
    print(f"\n{'='*60}")
    print(f"CHANNEL {ch} SETTINGS")
    print(f"{'='*60}\n")
    
    # Basic Info
    print("📋 BASIC INFO")
    print("-" * 60)
    print(f"  Name:        {format_value(f'/ch/{ch}/name', state.get(f'/ch/{ch}/name'))}")
    print(f"  Color:       {format_value(f'/ch/{ch}/col', state.get(f'/ch/{ch}/col'))}")
    print(f"  Icon:        {format_value(f'/ch/{ch}/icon', state.get(f'/ch/{ch}/icon'))}")
    print()
    
    # Channel Controls
    print("🎛️  CHANNEL CONTROLS")
    print("-" * 60)
    mute = state.get(f'/ch/{ch}/mute')
    mute_str = "MUTED" if mute == 1 else "UNMUTED" if mute == 0 else "N/A"
    print(f"  Mute:        {mute_str}")
    print(f"  Fader:       {format_value(f'/ch/{ch}/fdr', state.get(f'/ch/{ch}/fdr'))} dB")
    print(f"  Pan:         {format_value(f'/ch/{ch}/pan', state.get(f'/ch/{ch}/pan'))}")
    print(f"  Width:       {format_value(f'/ch/{ch}/wid', state.get(f'/ch/{ch}/wid'))} %")
    print()
    
    # Input Settings
    print("🔌 INPUT SETTINGS")
    print("-" * 60)
    print(f"  Trim:        {format_value(f'/ch/{ch}/in/set/trim', state.get(f'/ch/{ch}/in/set/trim'))} dB")
    print(f"  Balance:     {format_value(f'/ch/{ch}/in/set/bal', state.get(f'/ch/{ch}/in/set/bal'))} dB")
    inv = state.get(f'/ch/{ch}/in/set/inv')
    inv_str = "INVERTED" if inv == 1 else "NORMAL" if inv == 0 else "N/A"
    print(f"  Phase:       {inv_str}")
    print(f"  Delay Mode:  {format_value(f'/ch/{ch}/in/set/dlymode', state.get(f'/ch/{ch}/in/set/dlymode'))}")
    dlyon = state.get(f'/ch/{ch}/in/set/dlyon')
    dlyon_str = "ON" if dlyon == 1 else "OFF" if dlyon == 0 else "N/A"
    print(f"  Delay:       {format_value(f'/ch/{ch}/in/set/dly', state.get(f'/ch/{ch}/in/set/dly'))} ({dlyon_str})")
    print()
    
    # Filters
    print("🔊 FILTERS")
    print("-" * 60)
    lc = state.get(f'/ch/{ch}/flt/lc')
    lc_str = "ON" if lc == 1 else "OFF" if lc == 0 else "N/A"
    print(f"  Low Cut:     {lc_str} @ {format_value(f'/ch/{ch}/flt/lcf', state.get(f'/ch/{ch}/flt/lcf'))} Hz")
    print(f"  Low Cut Slope: {format_value(f'/ch/{ch}/flt/lcs', state.get(f'/ch/{ch}/flt/lcs'))} dB/oct")
    hc = state.get(f'/ch/{ch}/flt/hc')
    hc_str = "ON" if hc == 1 else "OFF" if hc == 0 else "N/A"
    print(f"  High Cut:    {hc_str} @ {format_value(f'/ch/{ch}/flt/hcf', state.get(f'/ch/{ch}/flt/hcf'))} Hz")
    print(f"  High Cut Slope: {format_value(f'/ch/{ch}/flt/hcs', state.get(f'/ch/{ch}/flt/hcs'))} dB/oct")
    print()
    
    # EQ
    print("🎚️  EQ")
    print("-" * 60)
    eq_on = state.get(f'/ch/{ch}/eq/on')
    eq_on_str = "ON" if eq_on == 1 else "OFF" if eq_on == 0 else "N/A"
    print(f"  EQ:          {eq_on_str}")
    if eq_on == 1:
        print(f"  Model:       {format_value(f'/ch/{ch}/eq/mdl', state.get(f'/ch/{ch}/eq/mdl'))}")
        print(f"  Mix:         {format_value(f'/ch/{ch}/eq/mix', state.get(f'/ch/{ch}/eq/mix'))} %")
        print(f"  Low Shelf:   {format_value(f'/ch/{ch}/eq/lg', state.get(f'/ch/{ch}/eq/lg'))} dB @ {format_value(f'/ch/{ch}/eq/lf', state.get(f'/ch/{ch}/eq/lf'))} Hz")
        print(f"  Band 1:      {format_value(f'/ch/{ch}/eq/1g', state.get(f'/ch/{ch}/eq/1g'))} dB @ {format_value(f'/ch/{ch}/eq/1f', state.get(f'/ch/{ch}/eq/1f'))} Hz")
        print(f"  Band 2:      {format_value(f'/ch/{ch}/eq/2g', state.get(f'/ch/{ch}/eq/2g'))} dB @ {format_value(f'/ch/{ch}/eq/2f', state.get(f'/ch/{ch}/eq/2f'))} Hz")
        print(f"  Band 3:      {format_value(f'/ch/{ch}/eq/3g', state.get(f'/ch/{ch}/eq/3g'))} dB @ {format_value(f'/ch/{ch}/eq/3f', state.get(f'/ch/{ch}/eq/3f'))} Hz")
        print(f"  Band 4:      {format_value(f'/ch/{ch}/eq/4g', state.get(f'/ch/{ch}/eq/4g'))} dB @ {format_value(f'/ch/{ch}/eq/4f', state.get(f'/ch/{ch}/eq/4f'))} Hz")
        print(f"  High Shelf:  {format_value(f'/ch/{ch}/eq/hg', state.get(f'/ch/{ch}/eq/hg'))} dB @ {format_value(f'/ch/{ch}/eq/hf', state.get(f'/ch/{ch}/eq/hf'))} Hz")
    print()
    
    # Dynamics
    print("⚡ DYNAMICS (COMPRESSOR)")
    print("-" * 60)
    dyn_on = state.get(f'/ch/{ch}/dyn/on')
    dyn_on_str = "ON" if dyn_on == 1 else "OFF" if dyn_on == 0 else "N/A"
    print(f"  Compressor:  {dyn_on_str}")
    if dyn_on == 1:
        print(f"  Model:       {format_value(f'/ch/{ch}/dyn/mdl', state.get(f'/ch/{ch}/dyn/mdl'))}")
        print(f"  Threshold:   {format_value(f'/ch/{ch}/dyn/thr', state.get(f'/ch/{ch}/dyn/thr'))} dB")
        print(f"  Ratio:       {format_value(f'/ch/{ch}/dyn/ratio', state.get(f'/ch/{ch}/dyn/ratio'))}")
        print(f"  Attack:      {format_value(f'/ch/{ch}/dyn/att', state.get(f'/ch/{ch}/dyn/att'))} ms")
        print(f"  Release:     {format_value(f'/ch/{ch}/dyn/rel', state.get(f'/ch/{ch}/dyn/rel'))} ms")
        print(f"  Gain:        {format_value(f'/ch/{ch}/dyn/gain', state.get(f'/ch/{ch}/dyn/gain'))} dB")
        print(f"  Mix:         {format_value(f'/ch/{ch}/dyn/mix', state.get(f'/ch/{ch}/dyn/mix'))} %")
    print()
    
    # Gate
    print("🚪 GATE")
    print("-" * 60)
    gate_on = state.get(f'/ch/{ch}/gate/on')
    gate_on_str = "ON" if gate_on == 1 else "OFF" if gate_on == 0 else "N/A"
    print(f"  Gate:        {gate_on_str}")
    if gate_on == 1:
        print(f"  Model:       {format_value(f'/ch/{ch}/gate/mdl', state.get(f'/ch/{ch}/gate/mdl'))}")
        print(f"  Threshold:   {format_value(f'/ch/{ch}/gate/thr', state.get(f'/ch/{ch}/gate/thr'))} dB")
        print(f"  Range:       {format_value(f'/ch/{ch}/gate/range', state.get(f'/ch/{ch}/gate/range'))} dB")
        print(f"  Attack:      {format_value(f'/ch/{ch}/gate/att', state.get(f'/ch/{ch}/gate/att'))} ms")
        print(f"  Release:     {format_value(f'/ch/{ch}/gate/rel', state.get(f'/ch/{ch}/gate/rel'))} ms")
    print()
    
    # Inserts
    print("🔌 INSERTS")
    print("-" * 60)
    
    # Pre Insert
    preins_on = state.get(f'/ch/{ch}/preins/on')
    preins_on_str = "ON" if preins_on == 1 else "OFF" if preins_on == 0 else "N/A"
    preins_slot = state.get(f'/ch/{ch}/preins/ins', 'NONE')
    preins_stat = state.get(f'/ch/{ch}/preins/$stat', 'N/A')
    print(f"  Pre Insert:  {preins_on_str}")
    if preins_on == 1 and preins_slot and preins_slot != 'NONE':
        print(f"    Slot:      {preins_slot}")
        print(f"    Status:    {preins_stat}")
    
    # Post Insert
    postins_on = state.get(f'/ch/{ch}/postins/on')
    postins_on_str = "ON" if postins_on == 1 else "OFF" if postins_on == 0 else "N/A"
    postins_slot = state.get(f'/ch/{ch}/postins/ins', 'NONE')
    postins_mode = state.get(f'/ch/{ch}/postins/mode', 'N/A')
    postins_stat = state.get(f'/ch/{ch}/postins/$stat', 'N/A')
    postins_w = state.get(f'/ch/{ch}/postins/w')
    print(f"  Post Insert: {postins_on_str}")
    if postins_on == 1 and postins_slot and postins_slot != 'NONE':
        print(f"    Slot:      {postins_slot}")
        print(f"    Mode:      {postins_mode}")
        print(f"    Weight:    {format_value(f'/ch/{ch}/postins/w', postins_w)}")
        print(f"    Status:    {postins_stat}")
    print()
    
    # Summary of all values
    print("📊 ALL RAW VALUES")
    print("-" * 60)
    channel_addresses = [addr for addr in state.keys() if addr.startswith(f'/ch/{ch}/')]
    if channel_addresses:
        for addr in sorted(channel_addresses):
            value = state.get(addr)
            print(f"  {addr:40s} = {value}")
    else:
        print("  No channel 10 data received")
    print()


def main():
    import sys
    
    logger.info("=== Channel 10 Settings Query ===")
    
    # Get connection info from command line or use defaults
    if len(sys.argv) >= 2:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])
    else:
        port = 2223
    
    channel = 10
    
    logger.info(f"\nConnecting to Wing at {ip}:{port}...")
    
    client = WingClient(ip, port)
    
    if not client.connect():
        logger.error("✗ Connection failed!")
        logger.error("Check:")
        logger.error("  1. Wing is powered on and connected to network")
        logger.error("  2. IP address is correct")
        logger.error("  3. Wing OSC is enabled")
        return
    
    logger.info("✓ Connected successfully!")
    logger.info(f"Querying channel {channel} settings...\n")
    
    # Get all addresses for channel 10
    addresses = get_all_channel_addresses(channel)
    
    # Query all addresses
    for addr in addresses:
        client.send(addr)
        time.sleep(0.01)  # Small delay between queries
    
    # Wait for responses
    logger.info("Waiting for responses...")
    time.sleep(2)
    
    # Get current state
    state = client.get_state()
    
    # Check for inserts and query FX modules if present
    preins_slot = state.get(f'/ch/{channel}/preins/ins')
    postins_slot = state.get(f'/ch/{channel}/postins/ins')
    
    fx_slots_to_query = []
    if preins_slot and preins_slot != 'NONE':
        fx_slots_to_query.append(preins_slot)
    if postins_slot and postins_slot != 'NONE':
        fx_slots_to_query.append(postins_slot)
    
    # Query FX modules if inserts are present
    if fx_slots_to_query:
        logger.info(f"Found inserts: {fx_slots_to_query}")
        logger.info("Querying FX module settings...")
        
        for fx_slot in fx_slots_to_query:
            fx_addresses = get_fx_addresses(fx_slot)
            # Also query the root node
            fx_num = fx_slot.replace('FX', '') if fx_slot.startswith('FX') else fx_slot
            client.send(f'/fx/{fx_num}/')  # Query root node
            time.sleep(0.05)
            
            for addr in fx_addresses:
                client.send(addr)
                time.sleep(0.01)
        
        # Wait for FX responses
        time.sleep(2)
        state = client.get_state()  # Update state with FX data
    
    # Print formatted settings
    print_channel_settings(channel, state)
    
    # Print FX module details if present
    if fx_slots_to_query:
        print_fx_modules(fx_slots_to_query, state)
    
    # Disconnect
    client.disconnect()
    logger.info("✓ Disconnected")


if __name__ == "__main__":
    main()
