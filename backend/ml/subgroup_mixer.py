"""
Hierarchical subgroup/bus mixing for automatic live concert mixing.

Implements a standard live mixing hierarchy:
  Individual channels -> Subgroup buses -> Master bus

Buses:
  drums_bus: kick, snare, hihat, toms, overheads, percussion
  guitar_bus: electric_guitar, acoustic_guitar
  vocal_bus: vocals
  keys_bus: keys
  bass_bus: bass_guitar
  aux_bus: brass, strings, unclassified

Each bus has per-subgroup processing (EQ, compression) before
being summed to the master output.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.signal import butter, lfilter

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Bus definitions: which instrument types belong to which bus
BUS_ASSIGNMENTS = {
    "drums_bus": ["kick", "snare", "hihat", "toms", "overheads", "percussion"],
    "guitar_bus": ["electric_guitar", "acoustic_guitar"],
    "vocal_bus": ["vocals"],
    "keys_bus": ["keys"],
    "bass_bus": ["bass_guitar"],
    "aux_bus": ["brass", "strings"],
}

# Default bus processing parameters
BUS_PROCESSING = {
    "drums_bus": {
        "hpf_freq": 40.0,      # Hz, high-pass filter
        "comp_threshold": -12.0, # dB
        "comp_ratio": 3.0,
        "comp_attack_ms": 5.0,
        "comp_release_ms": 80.0,
        "bus_gain_db": 0.0,
    },
    "guitar_bus": {
        "hpf_freq": 80.0,
        "comp_threshold": -15.0,
        "comp_ratio": 2.5,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 100.0,
        "bus_gain_db": -1.0,
    },
    "vocal_bus": {
        "hpf_freq": 100.0,
        "comp_threshold": -14.0,
        "comp_ratio": 3.0,
        "comp_attack_ms": 5.0,
        "comp_release_ms": 80.0,
        "bus_gain_db": 2.0,
    },
    "keys_bus": {
        "hpf_freq": 60.0,
        "comp_threshold": -16.0,
        "comp_ratio": 2.0,
        "comp_attack_ms": 10.0,
        "comp_release_ms": 120.0,
        "bus_gain_db": -1.0,
    },
    "bass_bus": {
        "hpf_freq": 30.0,
        "comp_threshold": -10.0,
        "comp_ratio": 4.0,
        "comp_attack_ms": 5.0,
        "comp_release_ms": 100.0,
        "bus_gain_db": 1.0,
    },
    "aux_bus": {
        "hpf_freq": 80.0,
        "comp_threshold": -18.0,
        "comp_ratio": 2.0,
        "comp_attack_ms": 15.0,
        "comp_release_ms": 150.0,
        "bus_gain_db": -2.0,
    },
}

# Master bus processing
MASTER_PROCESSING = {
    "comp_threshold": -8.0,
    "comp_ratio": 2.0,
    "comp_attack_ms": 10.0,
    "comp_release_ms": 150.0,
    "limiter_ceiling_db": -1.0,
}


class SubgroupMixer:
    """
    Hierarchical subgroup mixing engine.

    Routes classified channels to appropriate buses, applies per-bus
    processing, and sums to a stereo master output.
    """

    def __init__(self, sr=48000):
        """
        Args:
            sr: sample rate
        """
        self.sr = sr
        self.bus_assignments = dict(BUS_ASSIGNMENTS)
        self.bus_processing = dict(BUS_PROCESSING)
        self.master_processing = dict(MASTER_PROCESSING)

        # Current channel-to-bus mapping
        self._channel_bus_map = {}

    def assign_channels(self, channel_classifications):
        """
        Assign channels to buses based on their instrument classifications.

        Args:
            channel_classifications: dict mapping channel_id -> instrument_type
                e.g. {"ch1": "kick", "ch2": "snare", "ch5": "vocals"}

        Returns:
            dict mapping bus_name -> list of channel_ids
        """
        bus_channels = {bus: [] for bus in self.bus_assignments}

        for ch_id, instrument in channel_classifications.items():
            assigned = False
            for bus_name, instruments in self.bus_assignments.items():
                if instrument in instruments:
                    bus_channels[bus_name].append(ch_id)
                    self._channel_bus_map[ch_id] = bus_name
                    assigned = True
                    break
            if not assigned:
                # Unrecognized instrument goes to aux bus
                bus_channels["aux_bus"].append(ch_id)
                self._channel_bus_map[ch_id] = "aux_bus"
                logger.debug(
                    f"Channel {ch_id} ({instrument}) assigned to aux_bus (unrecognized)"
                )

        # Log assignments
        for bus_name, channels in bus_channels.items():
            if channels:
                logger.info(f"{bus_name}: {channels}")

        return bus_channels

    def get_bus_for_channel(self, channel_id):
        """Get the bus name a channel is assigned to."""
        return self._channel_bus_map.get(channel_id, "aux_bus")

    def compute_bus_mix(self, bus_name, channel_audios, channel_gains_db=None):
        """
        Mix multiple channels into a single bus output with bus processing.

        Args:
            bus_name: string bus name (e.g. "drums_bus")
            channel_audios: list of 1D numpy arrays (one per channel in bus)
            channel_gains_db: optional list of per-channel gain in dB

        Returns:
            bus_audio: 1D numpy array, processed bus output
        """
        if not channel_audios:
            return np.zeros(0)

        # Determine output length
        max_len = max(len(a) for a in channel_audios)

        # Sum channels with optional per-channel gain
        bus_sum = np.zeros(max_len, dtype=np.float64)
        for i, audio in enumerate(channel_audios):
            audio = np.asarray(audio, dtype=np.float64)
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            elif len(audio) > max_len:
                audio = audio[:max_len]

            if channel_gains_db is not None and i < len(channel_gains_db):
                gain = 10.0 ** (channel_gains_db[i] / 20.0)
                audio = audio * gain

            bus_sum += audio

        # Apply bus processing
        params = self.bus_processing.get(bus_name, self.bus_processing["aux_bus"])
        processed = self._apply_bus_processing(bus_sum, params)

        return processed

    def _apply_bus_processing(self, audio, params):
        """
        Apply bus-level processing chain: HPF -> Compression -> Bus Gain.

        Args:
            audio: 1D numpy array
            params: dict of processing parameters

        Returns:
            processed: 1D numpy array
        """
        if len(audio) == 0:
            return audio

        # High-pass filter
        hpf_freq = params.get("hpf_freq", 40.0)
        audio = self._apply_hpf(audio, hpf_freq)

        # Compression
        audio = self._apply_compression(
            audio,
            threshold_db=params.get("comp_threshold", -12.0),
            ratio=params.get("comp_ratio", 2.0),
            attack_ms=params.get("comp_attack_ms", 10.0),
            release_ms=params.get("comp_release_ms", 100.0),
        )

        # Bus gain
        bus_gain = 10.0 ** (params.get("bus_gain_db", 0.0) / 20.0)
        audio = audio * bus_gain

        return audio

    def _apply_hpf(self, audio, cutoff_freq):
        """Apply a 2nd-order Butterworth high-pass filter."""
        nyq = self.sr / 2.0
        norm_freq = cutoff_freq / nyq

        if norm_freq >= 1.0 or norm_freq <= 0.0:
            return audio

        if HAS_SCIPY:
            b, a = butter(2, norm_freq, btype="high")
            return lfilter(b, a, audio)
        else:
            # Simple first-order high-pass IIR filter (numpy fallback)
            rc = 1.0 / (2.0 * np.pi * cutoff_freq)
            dt = 1.0 / self.sr
            alpha = rc / (rc + dt)
            output = np.zeros_like(audio)
            output[0] = audio[0]
            for i in range(1, len(audio)):
                output[i] = alpha * (output[i - 1] + audio[i] - audio[i - 1])
            return output

    def _apply_compression(self, audio, threshold_db, ratio, attack_ms, release_ms):
        """Apply simple feed-forward compressor."""
        if ratio <= 1.0 or len(audio) == 0:
            return audio

        # RMS envelope with attack/release smoothing
        frame_size = max(1, int(self.sr * 0.002))  # 2ms frames
        n_frames = max(1, len(audio) // frame_size)

        # Compute per-frame RMS
        rms_frames = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            rms_frames[i] = np.sqrt(np.mean(audio[start:end] ** 2) + 1e-10)

        # Smooth envelope
        attack_coeff = np.exp(-1.0 / (attack_ms * self.sr / (1000.0 * frame_size) + 1e-8))
        release_coeff = np.exp(-1.0 / (release_ms * self.sr / (1000.0 * frame_size) + 1e-8))

        envelope = np.zeros(n_frames)
        envelope[0] = rms_frames[0]
        for i in range(1, n_frames):
            if rms_frames[i] > envelope[i - 1]:
                envelope[i] = attack_coeff * envelope[i - 1] + (1 - attack_coeff) * rms_frames[i]
            else:
                envelope[i] = release_coeff * envelope[i - 1] + (1 - release_coeff) * rms_frames[i]

        # Compute gain reduction
        envelope_db = 20.0 * np.log10(envelope + 1e-10)
        over_threshold = np.maximum(envelope_db - threshold_db, 0.0)
        gain_reduction_db = over_threshold * (1.0 - 1.0 / ratio)

        # Expand gain back to sample level
        gain_linear = np.zeros(len(audio))
        for i in range(n_frames):
            start = i * frame_size
            end = min(start + frame_size, len(audio))
            gain_linear[start:end] = 10.0 ** (-gain_reduction_db[i] / 20.0)

        return audio * gain_linear

    def compute_master(self, bus_audios):
        """
        Sum bus outputs and apply master bus processing.

        Args:
            bus_audios: dict mapping bus_name -> 1D numpy array
                or list of 1D numpy arrays

        Returns:
            master: 1D numpy array, final master output
        """
        if isinstance(bus_audios, dict):
            audios = [v for v in bus_audios.values() if len(v) > 0]
        else:
            audios = [a for a in bus_audios if len(a) > 0]

        if not audios:
            return np.zeros(0)

        max_len = max(len(a) for a in audios)
        master = np.zeros(max_len, dtype=np.float64)

        for audio in audios:
            if len(audio) < max_len:
                audio = np.pad(audio, (0, max_len - len(audio)))
            master += audio

        # Master bus compression
        master = self._apply_compression(
            master,
            threshold_db=self.master_processing["comp_threshold"],
            ratio=self.master_processing["comp_ratio"],
            attack_ms=self.master_processing["comp_attack_ms"],
            release_ms=self.master_processing["comp_release_ms"],
        )

        # Master limiter (brick-wall)
        ceiling = 10.0 ** (self.master_processing["limiter_ceiling_db"] / 20.0)
        peak = np.max(np.abs(master))
        if peak > ceiling:
            master = master * (ceiling / peak)

        return master

    def process_full_mix(self, channel_audios, channel_classifications,
                         channel_gains_db=None):
        """
        Full mixing pipeline: assign -> bus mix -> master.

        Args:
            channel_audios: dict mapping channel_id -> 1D numpy array
            channel_classifications: dict mapping channel_id -> instrument_type
            channel_gains_db: optional dict mapping channel_id -> gain in dB

        Returns:
            dict with keys:
                'master': 1D numpy array (master output)
                'buses': dict mapping bus_name -> 1D numpy array
                'assignments': dict mapping bus_name -> list of channel_ids
        """
        # Assign channels to buses
        assignments = self.assign_channels(channel_classifications)

        # Mix each bus
        bus_outputs = {}
        for bus_name, ch_ids in assignments.items():
            if not ch_ids:
                bus_outputs[bus_name] = np.zeros(0)
                continue

            bus_ch_audios = []
            bus_ch_gains = []
            for ch_id in ch_ids:
                if ch_id in channel_audios:
                    bus_ch_audios.append(channel_audios[ch_id])
                    if channel_gains_db and ch_id in channel_gains_db:
                        bus_ch_gains.append(channel_gains_db[ch_id])
                    else:
                        bus_ch_gains.append(0.0)

            gains = bus_ch_gains if channel_gains_db else None
            bus_outputs[bus_name] = self.compute_bus_mix(
                bus_name, bus_ch_audios, channel_gains_db=gains
            )

        # Master mix
        master = self.compute_master(bus_outputs)

        return {
            "master": master,
            "buses": bus_outputs,
            "assignments": assignments,
        }

    # ------------------------------------------------------------------
    # Convenience interface (assign_channel / get_group_settings /
    # compute_group_mix / get_bus_processing)
    # ------------------------------------------------------------------

    # Canonical group names and their mapping to internal bus names.
    _GROUP_TO_BUS = {
        "drums":   "drums_bus",
        "bass":    "bass_bus",
        "guitars": "guitar_bus",
        "keys":    "keys_bus",
        "vocals":  "vocal_bus",
    }

    def assign_channel(self, ch_id, group_name):
        """
        Assign a single channel to a named group (drums, bass, guitars,
        keys, vocals).

        Unrecognised group names are routed to ``aux_bus``.

        Args:
            ch_id: unique channel identifier (string or int).
            group_name: one of 'drums', 'bass', 'guitars', 'keys',
                'vocals', or a raw bus name such as 'aux_bus'.
        """
        bus = self._GROUP_TO_BUS.get(group_name.lower(), group_name)
        if bus not in self.bus_processing:
            bus = "aux_bus"
            logger.debug(
                "Group '%s' not recognized, routing channel %s to aux_bus",
                group_name, ch_id,
            )
        self._channel_bus_map[ch_id] = bus
        logger.debug("Channel %s assigned to %s", ch_id, bus)

    def get_group_settings(self, group_name):
        """
        Return the processing settings for a named group.

        Args:
            group_name: e.g. 'drums', 'bass', 'guitars', 'keys', 'vocals',
                or a raw bus name.

        Returns:
            dict of processing parameters (hpf_freq, comp_threshold,
            comp_ratio, comp_attack_ms, comp_release_ms, bus_gain_db),
            or None if the group is unknown.
        """
        bus = self._GROUP_TO_BUS.get(group_name.lower(), group_name)
        params = self.bus_processing.get(bus)
        if params is None:
            logger.warning("Unknown group/bus: %s", group_name)
        return dict(params) if params is not None else None

    def compute_group_mix(self, channel_features):
        """
        Compute per-group summed feature vectors from individual channel
        features and return recommended group-level gain offsets.

        This is a *feature-level* mix computation (not audio-domain):
        it examines numeric descriptors of each channel to decide how
        loud each group should be relative to the others.

        Args:
            channel_features: dict mapping channel_id to a dict with at
                least ``"rms_db"`` (float, RMS level in dB).  Optional
                keys: ``"spectral_centroid"`` (Hz), ``"crest_factor"``
                (dB), ``"instrument_type"`` (string).

        Returns:
            dict mapping group_name to a dict::

                {
                    "channels": [list of ch_ids],
                    "avg_rms_db": float,
                    "recommended_gain_db": float,
                }
        """
        # Bucket channels into groups
        group_channels = {}
        for ch_id, feats in channel_features.items():
            bus = self._channel_bus_map.get(ch_id, "aux_bus")
            group_channels.setdefault(bus, []).append((ch_id, feats))

        result = {}
        # Global reference RMS: median across all channels
        all_rms = [
            f.get("rms_db", -40.0) for f in channel_features.values()
        ]
        if all_rms:
            median_rms = float(np.median(all_rms))
        else:
            median_rms = -20.0

        for bus, items in group_channels.items():
            ch_ids = [cid for cid, _ in items]
            rms_values = np.array(
                [feats.get("rms_db", -40.0) for _, feats in items],
                dtype=np.float64,
            )
            avg_rms = float(np.mean(rms_values)) if len(rms_values) > 0 else -40.0

            # Recommended gain: bring the group's average RMS towards
            # the global median, then apply the bus gain offset.
            bus_gain_offset = self.bus_processing.get(
                bus, self.bus_processing["aux_bus"]
            ).get("bus_gain_db", 0.0)
            recommended = (median_rms - avg_rms) + bus_gain_offset

            # Resolve friendly group name from bus name
            friendly = bus
            for gname, bname in self._GROUP_TO_BUS.items():
                if bname == bus:
                    friendly = gname
                    break

            result[friendly] = {
                "channels": ch_ids,
                "avg_rms_db": round(avg_rms, 2),
                "recommended_gain_db": round(float(np.clip(recommended, -20, 12)), 2),
            }

        return result

    def get_bus_processing(self, group_name):
        """
        Return the bus processing chain description for a group.

        This is similar to :meth:`get_group_settings` but also includes
        the EQ curve parameters derived from the high-pass filter and
        master bus context.

        Args:
            group_name: group or bus name.

        Returns:
            dict with keys ``hpf_freq``, ``compression`` (sub-dict),
            ``group_level_db``, ``eq_curve`` (list of
            ``{freq, gain_db, q}`` dicts), or None if unknown.
        """
        bus = self._GROUP_TO_BUS.get(group_name.lower(), group_name)
        params = self.bus_processing.get(bus)
        if params is None:
            logger.warning("Unknown group/bus for bus processing: %s", group_name)
            return None

        # Build a simple EQ curve based on the HPF frequency
        hpf = params.get("hpf_freq", 40.0)
        eq_curve = [
            {"freq": hpf, "gain_db": -12.0, "q": 0.707},  # HPF rolloff
            {"freq": hpf * 2, "gain_db": -1.0, "q": 1.0},  # gentle slope
        ]

        return {
            "hpf_freq": hpf,
            "compression": {
                "threshold_db": params.get("comp_threshold", -12.0),
                "ratio": params.get("comp_ratio", 2.0),
                "attack_ms": params.get("comp_attack_ms", 10.0),
                "release_ms": params.get("comp_release_ms", 100.0),
            },
            "group_level_db": params.get("bus_gain_db", 0.0),
            "eq_curve": eq_curve,
        }

    # ------------------------------------------------------------------
    # Parameter mutation
    # ------------------------------------------------------------------

    def set_bus_param(self, bus_name, param_name, value):
        """
        Update a single bus processing parameter.

        Args:
            bus_name: string bus name
            param_name: parameter key (e.g. "comp_threshold")
            value: new value
        """
        if bus_name not in self.bus_processing:
            logger.warning(f"Unknown bus: {bus_name}")
            return
        self.bus_processing[bus_name][param_name] = value

    def set_master_param(self, param_name, value):
        """Update a master bus processing parameter."""
        self.master_processing[param_name] = value
