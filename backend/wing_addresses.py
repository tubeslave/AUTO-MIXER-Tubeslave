"""
WING OSC Addresses - Complete reference from WING Remote Protocols v3.0.5

All addresses use format /ch/{ch} where {ch} is channel number (1-40) without zero padding.
For compatibility, both /ch/1 and /ch/01 formats may work, but /ch/1 is the documented format.
"""

WING_OSC_ADDRESSES = {
    "xremote": "/xremote",
    
    # Channel addresses (channels 1-40)
    "channel": {
        # Basic channel info
        "name": "/ch/{ch}/name",
        "icon": "/ch/{ch}/icon",
        "color": "/ch/{ch}/col",
        "led": "/ch/{ch}/led",
        "tags": "/ch/{ch}/tags",
        "clink": "/ch/{ch}/clink",  # Custom link
        
        # Channel input settings
        "in": {
            "set": {
                "mode": "/ch/{ch}/in/set/$mode",  # [RO] M, ST, M/S
                "srcauto": "/ch/{ch}/in/set/srcauto",  # Auto source switch
                "altsrc": "/ch/{ch}/in/set/altsrc",  # Main/alt switch
                "inv": "/ch/{ch}/in/set/inv",  # Phase invert
                "trim": "/ch/{ch}/in/set/trim",  # Trim (dB) -18..18
                "bal": "/ch/{ch}/in/set/bal",  # Balance (dB) -9..9
                "g": "/ch/{ch}/in/set/$g",  # [RO] Gain (depends on source)
                "vph": "/ch/{ch}/in/set/$vph",  # [RO] Phantom power
                "dlymode": "/ch/{ch}/in/set/dlymode",  # Delay mode: M, FT, MS, SMP
                "dly": "/ch/{ch}/in/set/dly",  # Delay value
                "dlyon": "/ch/{ch}/in/set/dlyon",  # Delay on/off
            },
            "conn": {
                "grp": "/ch/{ch}/in/conn/grp",  # Input connection group
                "in": "/ch/{ch}/in/conn/in",  # Input connection index
                "altgrp": "/ch/{ch}/in/conn/altgrp",  # Alt input group
                "altin": "/ch/{ch}/in/conn/altin",  # Alt input index
            }
        },
        
        # Filters
        "flt": {
            "lc": "/ch/{ch}/flt/lc",  # Low cut switch (0/1)
            "lcf": "/ch/{ch}/flt/lcf",  # Low cut frequency (Hz) 20..2000
            "lcs": "/ch/{ch}/flt/lcs",  # Low cut slope: 6, 12, 18, 24
            "hc": "/ch/{ch}/flt/hc",  # High cut switch (0/1)
            "hcf": "/ch/{ch}/flt/hcf",  # High cut frequency (Hz) 50..20000
            "hcs": "/ch/{ch}/flt/hcs",  # High cut slope: 6, 12
            "tf": "/ch/{ch}/flt/tf",  # Tool filter switch
            "mdl": "/ch/{ch}/flt/mdl",  # Filter model: TILT, MAX, AP1, AP2
            "tilt": "/ch/{ch}/flt/tilt",  # Tilt level (dB) -6..6
        },
        
        # Channel controls
        "mute": "/ch/{ch}/mute",  # Mute (0/1)
        "fdr": "/ch/{ch}/fdr",  # Fader (dB) -144..10
        "pan": "/ch/{ch}/pan",  # Pan -100..100
        "wid": "/ch/{ch}/wid",  # Width (%) -150..150
        "solo": "/ch/{ch}/$solo",  # [RO] Solo switch
        "sololed": "/ch/{ch}/$sololed",  # [RO] Solo LED
        "solosafe": "/ch/{ch}/solosafe",  # Solo safe
        "mon": "/ch/{ch}/mon",  # Monitor mode: A, B, A+B
        "proc": "/ch/{ch}/proc",  # Process order
        "ptap": "/ch/{ch}/ptap",  # Pretap
        "presolo": "/ch/{ch}/$presolo",  # [RO] Presolo
        
        # PreSend EQ
        "peq": {
            "on": "/ch/{ch}/peq/on",
            "1g": "/ch/{ch}/peq/1g",  # Band 1 gain (dB) -15..15
            "1f": "/ch/{ch}/peq/1f",  # Band 1 frequency (Hz) 20..20000
            "1q": "/ch/{ch}/peq/1q",  # Band 1 Q 0.44..10
            "2g": "/ch/{ch}/peq/2g",
            "2f": "/ch/{ch}/peq/2f",
            "2q": "/ch/{ch}/peq/2q",
            "3g": "/ch/{ch}/peq/3g",
            "3f": "/ch/{ch}/peq/3f",
            "3q": "/ch/{ch}/peq/3q",
        },
        
        # Gate
        "gate": {
            "on": "/ch/{ch}/gate/on",
            "mdl": "/ch/{ch}/gate/mdl",  # Gate model
            "thr": "/ch/{ch}/gate/thr",  # Threshold (dB) -80..0
            "range": "/ch/{ch}/gate/range",  # Range (dB) 3..60
            "att": "/ch/{ch}/gate/att",  # Attack (ms) 0..120
            "hld": "/ch/{ch}/gate/hld",  # Hold (ms) 0..200
            "rel": "/ch/{ch}/gate/rel",  # Release (ms) 4..4000
            "acc": "/ch/{ch}/gate/acc",  # Accent (5) 0..100
            "ratio": "/ch/{ch}/gate/ratio",  # Ratio: 1:1.5, 1:2, 1:3, 1:4, GATE
            "sc": {
                "type": "/ch/{ch}/gatesc/type",  # Sidechain type: Off, LP12, HP12, BP
                "f": "/ch/{ch}/gatesc/f",  # Sidechain frequency (Hz) 20..20000
                "q": "/ch/{ch}/gatesc/q",  # Sidechain Q 0.44..10
                "src": "/ch/{ch}/gatesc/src",  # Sidechain source
                "tap": "/ch/{ch}/gatesc/tap",  # Sidechain tap
                "solo": "/ch/{ch}/gatesc/$solo",  # [RO] Sidechain solo
            }
        },
        
        # EQ
        "eq": {
            "on": "/ch/{ch}/eq/on",
            "mdl": "/ch/{ch}/eq/mdl",  # EQ model: STD, SOUL, E88, E84, F110, PULSAR, MACH4
            "mix": "/ch/{ch}/eq/mix",  # EQ mix (%) 0..125
            "solo": "/ch/{ch}/eq/$solo",  # [RO] EQ solo
            "solobd": "/ch/{ch}/eq/$solobd",  # [RO] EQ solo band
            # Low shelf
            "lg": "/ch/{ch}/eq/lg",  # Low gain (dB) -15..15
            "lf": "/ch/{ch}/eq/lf",  # Low frequency (Hz) 20..2000
            "lq": "/ch/{ch}/eq/lq",  # Low Q 0.44..10
            "leq": "/ch/{ch}/eq/leq",  # Low type: PEQ, SHV
            # Bands 1-4
            "1g": "/ch/{ch}/eq/1g",  # Band 1 gain (dB) -15..15
            "1f": "/ch/{ch}/eq/1f",  # Band 1 frequency (Hz) 20..20000
            "1q": "/ch/{ch}/eq/1q",  # Band 1 Q 0.44..10
            "2g": "/ch/{ch}/eq/2g",
            "2f": "/ch/{ch}/eq/2f",
            "2q": "/ch/{ch}/eq/2q",
            "3g": "/ch/{ch}/eq/3g",
            "3f": "/ch/{ch}/eq/3f",
            "3q": "/ch/{ch}/eq/3q",
            "4g": "/ch/{ch}/eq/4g",
            "4f": "/ch/{ch}/eq/4f",
            "4q": "/ch/{ch}/eq/4q",
            # High shelf
            "hg": "/ch/{ch}/eq/hg",  # High gain (dB) -15..15
            "hf": "/ch/{ch}/eq/hf",  # High frequency (Hz) 50..20000
            "hq": "/ch/{ch}/eq/hq",  # High Q 0.44..10
            "heq": "/ch/{ch}/eq/heq",  # High type: SHV, PEQ
        },
        
        # Dynamics (Compressor)
        "dyn": {
            "on": "/ch/{ch}/dyn/on",
            "mdl": "/ch/{ch}/dyn/mdl",  # Compressor model
            "mix": "/ch/{ch}/dyn/mix",  # Mix (%) 0..100
            "gain": "/ch/{ch}/dyn/gain",  # Gain (dB) -6..12
            "thr": "/ch/{ch}/dyn/thr",  # Threshold (dB) -60..0
            "ratio": "/ch/{ch}/dyn/ratio",  # Ratio: 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10, 20, 50, 100
            "knee": "/ch/{ch}/dyn/knee",  # Knee 0..5
            "det": "/ch/{ch}/dyn/det",  # Detect: PEAK, RMS
            "att": "/ch/{ch}/dyn/att",  # Attack (ms) 0..120
            "hld": "/ch/{ch}/dyn/hld",  # Hold (ms) 1..200
            "rel": "/ch/{ch}/dyn/rel",  # Release (ms) 4..4000
            "env": "/ch/{ch}/dyn/env",  # Envelope: LIN, LOG
            "auto": "/ch/{ch}/dyn/auto",  # Auto switch
            "xo": {
                "depth": "/ch/{ch}/dynxo/depth",  # Crossover depth (dB) 0..20
                "type": "/ch/{ch}/dynxo/type",  # Crossover type: OFF, LO6, LO12, HI6, HI12, PC
                "f": "/ch/{ch}/dynxo/f",  # Crossover frequency (Hz) 20..20000
                "solo": "/ch/{ch}/dynxo/$solo",  # [RO] Crossover solo
            },
            "sc": {
                "type": "/ch/{ch}/dynsc/type",  # Sidechain type: Off, LP12, HP12, BP
                "f": "/ch/{ch}/dynsc/f",  # Sidechain frequency (Hz) 20..20000
                "q": "/ch/{ch}/dynsc/q",  # Sidechain Q 0.44..10
                "src": "/ch/{ch}/dynsc/src",  # Sidechain source: SELF, CH.1..CH.40
                "tap": "/ch/{ch}/dynsc/tap",  # Sidechain tap
                "solo": "/ch/{ch}/dynsc/$solo",  # [RO] Sidechain solo
            }
        },
        
        # Inserts
        "preins": {
            "on": "/ch/{ch}/preins/on",
            "ins": "/ch/{ch}/preins/ins",  # FX slot: NONE, FX1..FX16
            "stat": "/ch/{ch}/preins/$stat",  # [RO] Status: -, OK, N/A
        },
        "postins": {
            "on": "/ch/{ch}/postins/on",
            "mode": "/ch/{ch}/postins/mode",  # Mode: FX, AUTO_X, AUTO_Y
            "ins": "/ch/{ch}/postins/ins",  # FX slot: NONE, FX1..FX16
            "w": "/ch/{ch}/postins/w",  # Autogain weight -12..12
            "stat": "/ch/{ch}/postins/$stat",  # [RO] Status
        },
        
        # Main sends
        "main": {
            "pre": "/ch/{ch}/main/pre",  # Pre fader to Main
            "1": {
                "on": "/ch/{ch}/main/1/on",
                "lvl": "/ch/{ch}/main/1/lvl",  # Level (dB) -144..10
            },
            # Main 2-4 follow same pattern
        },
        
        # Sends (to buses)
        "send": {
            "1": {
                "on": "/ch/{ch}/send/1/on",
                "lvl": "/ch/{ch}/send/1/lvl",  # Level (dB) -144..10
                "pon": "/ch/{ch}/send/1/pon",  # Pre always on
                "mode": "/ch/{ch}/send/1/mode",  # Mode: PRE, POST, GRP
                "plink": "/ch/{ch}/send/1/plink",  # Pan link
                "pan": "/ch/{ch}/send/1/pan",  # Pan -100..100
            },
            # Sends 2-16 follow same pattern
            # Matrix sends: /ch/{ch}/send/MX<x>/on, /ch/{ch}/send/MX<x>/lvl, etc.
        },
        
        "tapwid": "/ch/{ch}/tapwid",  # Width
        "fdr_ro": "/ch/{ch}/$fdr",  # [RO] Fader level affected by DCA
        "mute_ro": "/ch/{ch}/$mute",  # [RO] Mute state
        "muteovr": "/ch/{ch}/$muteovr",  # Mute override
    },
    
    # Aux channels (1-8)
    "aux": {
        "in": {
            "set": {
                "trim": "/aux/{aux}/in/set/trim",  # Trim (dB) -18..18
                "bal": "/aux/{aux}/in/set/bal",  # Balance (dB) -9..9
                "inv": "/aux/{aux}/in/set/inv",  # Phase invert
            }
        },
        "mute": "/aux/{aux}/mute",
        "fdr": "/aux/{aux}/fdr",
        "pan": "/aux/{aux}/pan",
        "wid": "/aux/{aux}/wid",
        "eq": {
            "on": "/aux/{aux}/eq/on",
            # Similar structure to channel EQ
        },
        "dyn": {
            "on": "/aux/{aux}/dyn/on",
            "thr": "/aux/{aux}/dyn/thr",
            "depth": "/aux/{aux}/dyn/depth",
        }
    },
    
    # Bus channels (1-16)
    "bus": {
        "in": {
            "set": {
                "trim": "/bus/{bus}/in/set/trim",
                "bal": "/bus/{bus}/in/set/bal",
                "inv": "/bus/{bus}/in/set/inv",
            }
        },
        "mute": "/bus/{bus}/mute",
        "fdr": "/bus/{bus}/fdr",
        "pan": "/bus/{bus}/pan",
        "wid": "/bus/{bus}/wid",
        "busmono": "/bus/{bus}/busmono",  # Mono switch
        "eq": {
            "on": "/bus/{bus}/eq/on",
            # 8-band EQ structure similar to channel
        },
        "dyn": {
            "on": "/bus/{bus}/dyn/on",
            # Similar structure to channel dynamics
        },
        "dly": {
            "on": "/bus/{bus}/dly/on",
            "mode": "/bus/{bus}/dly/mode",
            "dly": "/bus/{bus}/dly/dly",
        }
    },
    
    # Main outputs (1-4)
    "main": {
        "mute": "/main/{main}/mute",
        "fdr": "/main/{main}/fdr",
        "pan": "/main/{main}/pan",
        "wid": "/main/{main}/wid",
        "eq": {
            "on": "/main/{main}/eq/on",
            # 8-band EQ structure
        },
        "dyn": {
            "on": "/main/{main}/dyn/on",
            # Similar structure
        }
    },
    
    # Matrix outputs (1-8)
    "mtx": {
        "mute": "/mtx/{mtx}/mute",
        "fdr": "/mtx/{mtx}/fdr",
        "pan": "/mtx/{mtx}/pan",
        "wid": "/mtx/{mtx}/wid",
        "eq": {
            "on": "/mtx/{mtx}/eq/on",
        }
    },
    
    # DCA groups (1-16)
    "dca": {
        "fdr": "/dca/{dca}/fdr",
        "mute": "/dca/{dca}/mute",
    }
}


def get_address(path: str, **kwargs) -> str:
    """
    Get OSC address from path string
    
    Args:
        path: Dot-separated path (e.g., "channel.in.set.trim")
        **kwargs: Format parameters (e.g., ch=1, aux=1, bus=1, main=1, mtx=1, dca=1)
    
    Returns:
        Formatted OSC address string
    
    Example:
        get_address("channel.in.set.trim", ch=1) -> "/ch/1/in/set/trim"
    """
    keys = path.split('.')
    current = WING_OSC_ADDRESSES
    
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                raise KeyError(f"Path '{path}' not found at key '{key}'")
        else:
            raise KeyError(f"Path '{path}' invalid - '{key}' is not a dict")
    
    if isinstance(current, str):
        return current.format(**kwargs)
    
    return current


def get_channel_address(channel: int, param: str) -> str:
    """
    Get channel OSC address
    
    Args:
        channel: Channel number (1-40)
        param: Dot-separated parameter path (e.g., "in.set.trim", "eq.1f", "gate.on")
    
    Returns:
        Formatted OSC address string
    
    Example:
        get_channel_address(1, "in.set.trim") -> "/ch/1/in/set/trim"
        get_channel_address(1, "eq.1f") -> "/ch/1/eq/1f"
    """
    parts = param.split('.')
    address_template = WING_OSC_ADDRESSES["channel"]
    
    for part in parts:
        if isinstance(address_template, dict):
            address_template = address_template.get(part)
            if address_template is None:
                raise KeyError(f"Channel parameter '{param}' not found at '{part}'")
        else:
            raise KeyError(f"Channel parameter '{param}' invalid - '{part}' is not a dict")
    
    if isinstance(address_template, str):
        return address_template.format(ch=channel)
    
    raise ValueError(f"Channel parameter '{param}' does not resolve to a string address")


def format_channel_number(channel: int, zero_padded: bool = False) -> str:
    """
    Format channel number for OSC address
    
    Args:
        channel: Channel number (1-40)
        zero_padded: Whether to zero-pad (01 vs 1)
    
    Returns:
        Formatted channel string
    """
    if zero_padded:
        return f"{channel:02d}"
    return str(channel)
