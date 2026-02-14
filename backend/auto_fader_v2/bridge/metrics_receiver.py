"""
Metrics Receiver - Receives and processes metrics from C++ core
"""

import logging
import threading
import time
from typing import Dict, Callable, Optional
from .cpp_bridge import CppBridge, ChannelMetrics

logger = logging.getLogger(__name__)


class MetricsReceiver:
    """
    Continuously receives metrics from C++ core and notifies listeners
    """
    
    def __init__(self, bridge: CppBridge, update_interval: float = 0.1):
        """
        Args:
            bridge: C++ bridge instance
            update_interval: How often to poll for metrics (seconds)
        """
        self.bridge = bridge
        self.update_interval = update_interval
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.metrics_callbacks: list[Callable[[Dict[int, ChannelMetrics]], None]] = []
    
    def add_callback(self, callback: Callable[[Dict[int, ChannelMetrics]], None]):
        """Add a callback to be called when new metrics arrive"""
        self.metrics_callbacks.append(callback)
    
    def start(self):
        """Start receiving metrics"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        logger.info("Metrics receiver started")
    
    def stop(self):
        """Stop receiving metrics"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Metrics receiver stopped")
    
    def _receive_loop(self):
        """Main receive loop"""
        loop_count = 0
        while self.running:
            try:
                # Read new metrics
                new_metrics = self.bridge.read_metrics()
                
                if new_metrics:
                    logger.debug(f"Received {len(new_metrics)} new metrics")
                    # Get all latest metrics
                    all_metrics = self.bridge.get_all_metrics()
                    
                    if all_metrics:
                        logger.debug(f"Total metrics available: {len(all_metrics)} channels")
                        # Log sample metrics every 10 loops (once per second)
                        if loop_count % 10 == 0:
                            sample_ch = list(all_metrics.keys())[0] if all_metrics else None
                            if sample_ch:
                                m = all_metrics[sample_ch]
                                logger.info(f"Sample metrics: channel={sample_ch}, lufs={m.lufs_momentary:.2f}, active={m.is_active}")
                                # Log all channel IDs to debug
                                channel_ids = list(all_metrics.keys())[:10]  # First 10
                                logger.debug(f"Channel IDs in metrics: {channel_ids}")
                    
                    # Notify callbacks
                    logger.debug(f"Calling {len(self.metrics_callbacks)} callbacks with {len(all_metrics)} metrics")
                    for callback in self.metrics_callbacks:
                        try:
                            callback(all_metrics)
                        except Exception as e:
                            logger.error(f"Error in metrics callback: {e}", exc_info=True)
                else:
                    # Log every 50 loops (every 5 seconds) if no metrics
                    if loop_count % 50 == 0:
                        logger.debug("No new metrics received")
                
                loop_count += 1
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics receive loop: {e}", exc_info=True)
                time.sleep(1.0)
