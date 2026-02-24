"""
C++ Bridge - Communication layer between Python and C++ core

Provides low-latency access to audio metrics from the C++ DSP engine
via shared memory IPC.
"""

import mmap
import struct
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import subprocess
import os
import time
import platform
import ctypes
import ctypes.util
import sys
import json

logger = logging.getLogger(__name__)


class SharedMemoryWrapper:
    """Wrapper for shared memory mapped via ctypes.mmap on macOS"""
    def __init__(self, ptr: int, size: int):
        self.ptr = ptr
        self.size = size
        self._closed = False
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.size
            step = key.step or 1
            return bytes(ctypes.c_uint8 * (stop - start)).from_address(self.ptr + start)[::step]
        else:
            return ctypes.c_uint8.from_address(self.ptr + key).value
    
    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or self.size
            data = bytes(value)
            ctypes.memmove(self.ptr + start, data, len(data))
        else:
            ctypes.c_uint8.from_address(self.ptr + key).value = value
    
    def __len__(self):
        return self.size
    
    def read(self, size: int = -1, offset: int = 0):
        if size == -1:
            size = self.size - offset
        # Create array and read from memory
        arr = (ctypes.c_uint8 * size).from_address(self.ptr + offset)
        return bytes(arr)
    
    def write(self, data: bytes, offset: int = 0):
        ctypes.memmove(self.ptr + offset, data, len(data))
    
    def close(self):
        if not self._closed and self.ptr:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.munmap.restype = ctypes.c_int
            libc.munmap(self.ptr, self.size)
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass
class ChannelMetrics:
    """Audio metrics for a single channel (matches C++ struct)"""
    channel_id: int
    lufs_momentary: float
    lufs_short_term: float
    true_peak: float
    spectral_centroid: float
    spectral_flatness: float
    spectral_rolloff: float
    band_energy_sub: float
    band_energy_bass: float
    band_energy_low_mid: float
    band_energy_mid: float
    band_energy_high_mid: float
    band_energy_high: float
    band_energy_air: float
    is_active: bool
    rms_level: float
    timestamp_us: int


class CoreStatus(Enum):
    """C++ core status"""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    ERROR = 3


class CppBridge:
    """
    Bridge to C++ core for real-time audio processing
    
    Manages:
    - C++ process lifecycle
    - Shared memory communication
    - Metrics reception
    """
    
    # Shared memory layout constants (must match C++ SharedMemoryHeader)
    MAGIC_NUMBER = 0xAF020001
    VERSION = 1
    HEADER_SIZE = 32  # bytes
    # ChannelMetrics: int32_t (4) + 6 floats (24) + BandEnergy 7 floats (28) + bool (1) + padding (3) + float (4) + uint64_t (8) = 72 bytes
    METRICS_SIZE = 72  # bytes per ChannelMetrics
    
    def __init__(self, shm_name: str = "auto_fader_metrics"):
        self.shm_name = shm_name
        self.cpp_process: Optional[subprocess.Popen] = None
        self.shm_fd = None
        self.shm_mmap = None
        self.shm_file = None  # File object for Linux /dev/shm access
        self._temp_file = None  # Keep reference to file wrapper if used
        self.status = CoreStatus.STOPPED
        self.last_error = ""
        
        # Metrics cache
        self.latest_metrics: Dict[int, ChannelMetrics] = {}
        
        # Load POSIX shared memory functions
        if platform.system() == "Darwin":  # macOS
            self.libc = ctypes.CDLL(ctypes.util.find_library("c"))
            self.libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
            self.libc.shm_open.restype = ctypes.c_int
            self.libc.shm_unlink.argtypes = [ctypes.c_char_p]
            self.libc.shm_unlink.restype = ctypes.c_int
            self.libc.close.argtypes = [ctypes.c_int]
            self.libc.close.restype = ctypes.c_int
            # mmap function
            self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
            self.libc.mmap.restype = ctypes.c_void_p
            self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            self.libc.munmap.restype = ctypes.c_int
    
    def start_cpp_core(self, config: Optional[Dict] = None) -> bool:
        """
        Start the C++ core process
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if successful
        """
        try:
            cpp_executable = os.path.join(
                os.path.dirname(__file__),
                "../../native/build/auto_fader_core"
            )
            
            if not os.path.exists(cpp_executable):
                logger.error(f"C++ executable not found: {cpp_executable}")
                logger.error("Please build the C++ core first: cd backend/native && ./build.sh")
                self.last_error = "C++ core not built"
                self.status = CoreStatus.ERROR
                return False
            
            logger.info("Starting C++ core...")
            self.status = CoreStatus.STARTING
            
            # Start C++ process
            self.cpp_process = subprocess.Popen(
                [cpp_executable],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr to stdout
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Start thread to read C++ process output
            import threading
            def read_cpp_output():
                if self.cpp_process and self.cpp_process.stdout:
                    for line in iter(self.cpp_process.stdout.readline, ''):
                        if line:
                            logger.info(f"C++ core: {line.strip()}")
            
            output_thread = threading.Thread(target=read_cpp_output, daemon=True)
            output_thread.start()
            
            # Wait for shared memory to be created
            timeout = 5.0
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.connect_shared_memory():
                    logger.info("C++ core started successfully")
                    self.status = CoreStatus.RUNNING
                    return True
                time.sleep(0.1)
            
            logger.error("Timeout waiting for C++ core to start")
            self.last_error = "Startup timeout"
            self.status = CoreStatus.ERROR
            return False
            
        except Exception as e:
            logger.error(f"Failed to start C++ core: {e}")
            self.last_error = str(e)
            self.status = CoreStatus.ERROR
            return False
    
    def stop_cpp_core(self):
        """Stop the C++ core process"""
        if self.cpp_process:
            logger.info("Stopping C++ core...")
            self.cpp_process.terminate()
            try:
                self.cpp_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("C++ core did not terminate, killing...")
                self.cpp_process.kill()
            self.cpp_process = None
        
        self.disconnect_shared_memory()
        self.status = CoreStatus.STOPPED
    
    def connect_shared_memory(self) -> bool:
        """
        Connect to shared memory region using POSIX shm_open
        
        Returns:
            True if successful
        """
        try:
            if platform.system() == "Darwin":  # macOS
                # Use POSIX shm_open
                shm_name_bytes = f"/{self.shm_name}".encode('utf-8')
                O_RDWR = 2
                
                # Try to open existing shared memory (O_RDWR only, no O_CREAT)
                self.shm_fd = self.libc.shm_open(shm_name_bytes, O_RDWR, 0o666)
                
                if self.shm_fd == -1:
                    # Shared memory doesn't exist yet
                    import errno
                    err = ctypes.get_errno()
                    logger.debug(f"shm_open failed: {os.strerror(err)} (errno: {err})")
                    return False
                
                # Get size of shared memory using fstat
                try:
                    st = os.fstat(self.shm_fd)
                    shm_size = st.st_size
                    logger.debug(f"Shared memory size from fstat: {shm_size} bytes")
                    
                    if shm_size == 0:
                        logger.error("Shared memory size is 0")
                        self.libc.close(self.shm_fd)
                        self.shm_fd = None
                        return False
                except Exception as e:
                    logger.error(f"Failed to get shared memory size: {e}")
                    self.libc.close(self.shm_fd)
                    self.shm_fd = None
                    return False
                
                # Map shared memory using mmap
                # On macOS, Python's mmap.mmap doesn't work well with shm_open FDs
                # Use direct ctypes.mmap call instead
                try:
                    PROT_READ = 1
                    PROT_WRITE = 2
                    MAP_SHARED = 1
                    
                    # Use ctypes to call mmap directly
                    mmap_ptr = self.libc.mmap(
                        None,  # Let system choose address
                        shm_size,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED,
                        self.shm_fd,
                        0  # offset
                    )
                    
                    # Check for error (mmap returns MAP_FAILED which is -1 on most systems)
                    if mmap_ptr == -1 or mmap_ptr == 0xFFFFFFFFFFFFFFFF:
                        raise OSError(f"mmap failed: {os.strerror(ctypes.get_errno())}")
                    
                    # Create wrapper for mmap'd memory
                    self.shm_mmap = SharedMemoryWrapper(mmap_ptr, shm_size)
                    logger.debug(f"Successfully mapped {shm_size} bytes using ctypes.mmap")
                except Exception as e:
                    logger.error(f"Failed to mmap shared memory: {e}")
                    self.libc.close(self.shm_fd)
                    self.shm_fd = None
                    return False
                
            else:  # Linux
                # Linux uses /dev/shm
                shm_path = f"/dev/shm/{self.shm_name}"
                if not os.path.exists(shm_path):
                    return False
                
                self.shm_file = open(shm_path, "r+b")
                self.shm_mmap = mmap.mmap(self.shm_file.fileno(), 0)
            
            # Validate header
            if not self._validate_header():
                logger.error("Invalid shared memory header")
                self.disconnect_shared_memory()
                return False
            
            logger.info(f"Connected to shared memory: {self.shm_name}")
            return True
            
        except Exception as e:
            logger.error(f"Could not connect to shared memory: {e}", exc_info=True)
            return False
    
    def disconnect_shared_memory(self):
        """Disconnect from shared memory"""
        if self.shm_mmap:
            self.shm_mmap.close()
            self.shm_mmap = None
        
        if self._temp_file:
            self._temp_file.close()
            self._temp_file = None
        
        if platform.system() == "Darwin":
            if self.shm_fd is not None:
                self.libc.close(self.shm_fd)
                self.shm_fd = None
        else:
            # Linux: close file if it exists
            if self.shm_file:
                self.shm_file.close()
                self.shm_file = None
    
    def _validate_header(self) -> bool:
        """Validate shared memory header"""
        if not self.shm_mmap or len(self.shm_mmap) < self.HEADER_SIZE:
            return False
        
        # Read magic number and version
        # Handle both mmap.mmap objects and SharedMemoryWrapper
        if isinstance(self.shm_mmap, SharedMemoryWrapper):
            header_data = self.shm_mmap.read(self.HEADER_SIZE, 0)
            magic, version = struct.unpack_from("II", header_data, 0)
        else:
            magic, version = struct.unpack_from("II", self.shm_mmap, 0)
        
        return magic == self.MAGIC_NUMBER and version == self.VERSION
    
    def read_metrics(self) -> List[ChannelMetrics]:
        """
        Read all available metrics from shared memory
        
        Returns:
            List of channel metrics
        """
        if not self.shm_mmap:
            return []
        
        try:
            # Read header - handle both mmap.mmap and SharedMemoryWrapper
            # Header: magic (4), version (4), num_channels (4), buffer_capacity (4), write_index (4), read_index (4), producer_alive (1), consumer_alive (1), padding (10)
            if isinstance(self.shm_mmap, SharedMemoryWrapper):
                header_bytes = self.shm_mmap.read(32, 0)
                # Unpack: 4 uint32_t + 2 bools (but struct.unpack treats bools as bytes, so use B for bools)
                magic, version, num_channels, buffer_capacity, write_idx, read_idx, producer_alive, consumer_alive = struct.unpack_from("IIIIIIBB", header_bytes, 0)
            else:
                magic, version, num_channels, buffer_capacity, write_idx, read_idx, producer_alive, consumer_alive = struct.unpack_from("IIIIIIBB", self.shm_mmap, 0)
            
            # Log buffer state periodically
            if hasattr(self, '_read_count'):
                self._read_count += 1
            else:
                self._read_count = 0
            
            if self._read_count % 10 == 0:  # Every 10 reads (once per second)
                logger.debug(f"Shared memory buffer: read_idx={read_idx}, write_idx={write_idx}, capacity={buffer_capacity}, producer_alive={bool(producer_alive)}")
            
            # Read available metrics
            metrics_list = []
            
            if read_idx == write_idx:
                # Buffer is empty
                if self._read_count % 50 == 0:  # Log every 5 seconds
                    logger.debug("Metrics buffer is empty (read_idx == write_idx)")
            else:
                logger.debug(f"Reading metrics: read_idx={read_idx}, write_idx={write_idx}, available={write_idx - read_idx if write_idx > read_idx else buffer_capacity - read_idx + write_idx}")
            
            while read_idx != write_idx:
                # Calculate base offset for this metric block
                base_offset = self.HEADER_SIZE + (read_idx % buffer_capacity) * self.METRICS_SIZE
                
                # FIX: Read structure from base_offset (correct position), but channel_id is 4 bytes BEFORE
                # So we read channel_id separately from base_offset - 4, then read rest of structure from base_offset
                read_offset = base_offset  # Read structure from here (where lufs_momentary starts)
                channel_id_offset = base_offset - 4  # channel_id is 4 bytes before
                
                # Safety check
                if channel_id_offset < 0:
                    channel_id_offset = 0
                
                # Read full structure from channel_id_offset (where channel_id actually is)
                try:
                    # Full format: int32_t + 6 floats + 7 floats (BandEnergy) + bool + padding(3) + float + uint64_t
                    struct_format = "i" + "f" * 6 + "f" * 7 + "?" + "xxx" + "f" + "Q"  # padding after bool
                    
                    if isinstance(self.shm_mmap, SharedMemoryWrapper):
                        # Read full 72 bytes starting from channel_id_offset
                        metrics_bytes = self.shm_mmap.read(self.METRICS_SIZE, channel_id_offset)
                        # Unpack with offset=4 to skip channel_id
                        metrics_data = struct.unpack_from(struct_format, metrics_bytes, 0)
                    else:
                        # Read full 72 bytes starting from channel_id_offset
                        metrics_bytes = self.shm_mmap[channel_id_offset:channel_id_offset+self.METRICS_SIZE]
                        # Unpack with offset=0 (channel_id is first)
                        metrics_data = struct.unpack_from(struct_format, metrics_bytes, 0)
                except Exception as e:
                    logger.error(f"Error reading metrics structure: {e}")
                    read_idx = (read_idx + 1) % buffer_capacity
                    continue
                
                # C++ uses 0-based channel IDs (0-31), but Python/Frontend expects 1-based (1-32)
                # Add 1 to channel_id for correct mapping
                metrics = ChannelMetrics(
                    channel_id=metrics_data[0] + 1,
                    lufs_momentary=metrics_data[1],
                    lufs_short_term=metrics_data[2],
                    true_peak=metrics_data[3],
                    spectral_centroid=metrics_data[4],
                    spectral_flatness=metrics_data[5],
                    spectral_rolloff=metrics_data[6],
                    band_energy_sub=metrics_data[7],
                    band_energy_bass=metrics_data[8],
                    band_energy_low_mid=metrics_data[9],
                    band_energy_mid=metrics_data[10],
                    band_energy_high_mid=metrics_data[11],
                    band_energy_high=metrics_data[12],
                    band_energy_air=metrics_data[13],
                    is_active=metrics_data[14],
                    rms_level=metrics_data[15],
                    timestamp_us=metrics_data[16]
                )
                
                metrics_list.append(metrics)
                self.latest_metrics[metrics.channel_id] = metrics
                
                # Update read index
                read_idx = (read_idx + 1) % buffer_capacity
            
            # Write back updated read index
            read_idx_bytes = struct.pack("I", read_idx)
            if isinstance(self.shm_mmap, SharedMemoryWrapper):
                self.shm_mmap.write(read_idx_bytes, 20)
            else:
                struct.pack_into("I", self.shm_mmap, 20, read_idx)
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Error reading metrics: {e}", exc_info=True)
            return []
    
    def get_channel_metrics(self, channel_id: int) -> Optional[ChannelMetrics]:
        """Get latest metrics for a specific channel"""
        return self.latest_metrics.get(channel_id)
    
    def get_all_metrics(self) -> Dict[int, ChannelMetrics]:
        """Get all latest metrics"""
        return self.latest_metrics.copy()
    
    def is_running(self) -> bool:
        """Check if C++ core is running"""
        return self.status == CoreStatus.RUNNING and self.cpp_process is not None
    
    def get_status(self) -> Dict:
        """Get bridge status"""
        return {
            "status": self.status.name,
            "cpp_running": self.is_running(),
            "shm_connected": self.shm_mmap is not None,
            "metrics_count": len(self.latest_metrics),
            "last_error": self.last_error
        }
    
    def test_connection(self) -> bool:
        """Test connection to C++ core"""
        try:
            logger.info("Testing C++ bridge connection...")
            
            # Try to connect to shared memory
            if self.connect_shared_memory():
                logger.info("✓ Shared memory connection successful")
                self.disconnect_shared_memory()
                return True
            else:
                logger.error("✗ Could not connect to shared memory")
                logger.error("Make sure C++ core is running")
                return False
                
        except Exception as e:
            logger.error(f"✗ Connection test failed: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_cpp_core()
