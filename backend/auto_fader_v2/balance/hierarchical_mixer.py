"""Priority-based hierarchical mixing for overload protection."""

from typing import Dict, List


class HierarchicalMixer:
    """
    Adds extra attenuation to low-priority channels when mix is overloaded.

    Priorities:
    - lower number = more important channel (e.g. vocal = 1)
    - higher number = less important channel
    """

    def __init__(
        self,
        enabled: bool = False,
        overload_threshold_lufs: float = -18.0,
        max_step_cut_db: float = 3.0,
        cut_scale: float = 0.6,
    ):
        self.enabled = enabled
        self.overload_threshold_lufs = overload_threshold_lufs
        self.max_step_cut_db = max_step_cut_db
        self.cut_scale = cut_scale

    def get_cuts(
        self,
        current_levels: Dict[int, float],
        selected_channels: List[int],
        reference_channels: List[int],
        channel_priorities: Dict[int, int],
    ) -> Dict[int, float]:
        """
        Returns extra cuts (negative dB values) by channel.

        The overload estimate uses average active level of selected non-reference channels.
        """
        if not self.enabled:
            return {}

        active_channels = [
            ch for ch in selected_channels if ch in current_levels and ch not in reference_channels
        ]
        if len(active_channels) < 2:
            return {}

        avg_level = sum(current_levels[ch] for ch in active_channels) / len(active_channels)
        overload_db = avg_level - self.overload_threshold_lufs
        if overload_db <= 0:
            return {}

        # Start cutting from least important channels first.
        sorted_by_priority = sorted(
            active_channels, key=lambda ch: channel_priorities.get(ch, 100), reverse=True
        )

        base_cut = min(self.max_step_cut_db, overload_db * self.cut_scale)
        cuts: Dict[int, float] = {}
        for index, ch in enumerate(sorted_by_priority):
            # Progressive attenuation: strongest cut on lowest priority.
            factor = max(0.25, 1.0 - (index * 0.12))
            cuts[ch] = -abs(base_cut * factor)

        return cuts
