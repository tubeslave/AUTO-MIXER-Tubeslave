from lufs_gain_staging import SafeGainCalibrator


def _new_calibrator() -> SafeGainCalibrator:
    cal = SafeGainCalibrator(mixer_client=None, sample_rate=48000, config={}, bleed_service=None)
    cal.add_channel(12)
    cal.add_channel(13)
    return cal


def test_stereo_pair_uses_average_minus_6_dbfs_for_duplicated_preset_name():
    cal = _new_calibrator()
    cal.channel_settings = {
        12: {"preset": "playback"},
        13: {"preset": "Playback"},
    }

    # Simulate already computed per-channel corrections:
    # ch12 corrected peak = -15 dBFS, ch13 corrected peak = -17 dBFS.
    cal.channels[12].max_true_peak_db = -20.0
    cal.channels[12].suggested_gain_db = 5.0
    cal.channels[13].max_true_peak_db = -20.0
    cal.channels[13].suggested_gain_db = 3.0

    cal._apply_stereo_pair_rule()

    # Average is -16 dBFS, then mono compensation = -6 dB => -22 dBFS per channel.
    assert cal.channels[12].target_peak_dbfs == -22.0
    assert cal.channels[13].target_peak_dbfs == -22.0
    assert cal.channels[12].suggested_gain_db == -2.0
    assert cal.channels[13].suggested_gain_db == -2.0
    assert cal.channels[12].gain_limited_by == "stereo_pair_balance"
    assert cal.channels[13].gain_limited_by == "stereo_pair_balance"


def test_stereo_pair_applies_for_ambience_preset():
    """Ambience is not in STEREO_PAIR_DRUM_EXCLUDED_PREFIXES — L/R share stereo balance."""
    cal = _new_calibrator()
    cal.channel_settings = {
        12: {"preset": "ambience"},
        13: {"preset": "Ambience"},
    }

    cal.channels[12].max_true_peak_db = -30.0
    cal.channels[12].suggested_gain_db = 2.0
    cal.channels[13].max_true_peak_db = -30.0
    cal.channels[13].suggested_gain_db = 0.0

    cal._apply_stereo_pair_rule()

    # Average corrected (-28 + -30) / 2 = -29, minus 6 dB mono comp => -35 dBFS target each.
    assert cal.channels[12].target_peak_dbfs == -35.0
    assert cal.channels[13].target_peak_dbfs == -35.0
    assert cal.channels[12].suggested_gain_db == -5.0
    assert cal.channels[13].suggested_gain_db == -5.0
    assert cal.channels[12].gain_limited_by == "stereo_pair_balance"
    assert cal.channels[13].gain_limited_by == "stereo_pair_balance"


def test_stereo_rule_is_not_applied_for_drum_parts():
    cal = _new_calibrator()
    cal.channel_settings = {
        12: {"preset": "Kick"},
        13: {"preset": "kick"},
    }

    cal.channels[12].max_true_peak_db = -20.0
    cal.channels[12].suggested_gain_db = 5.0
    cal.channels[13].max_true_peak_db = -20.0
    cal.channels[13].suggested_gain_db = 3.0

    cal._apply_stereo_pair_rule()

    assert cal.channels[12].target_peak_dbfs == -12.0
    assert cal.channels[13].target_peak_dbfs == -12.0
    assert cal.channels[12].suggested_gain_db == 5.0
    assert cal.channels[13].suggested_gain_db == 3.0
