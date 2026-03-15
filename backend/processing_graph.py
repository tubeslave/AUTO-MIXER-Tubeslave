"""
Audio processing graph — defines the signal flow for the mixing pipeline.
Each node represents a processing stage (EQ, compressor, gate, etc.)
connected in a directed acyclic graph.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class NodeType(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    GAIN = 'gain'
    EQ = 'eq'
    COMPRESSOR = 'compressor'
    GATE = 'gate'
    DELAY = 'delay'
    PAN = 'pan'
    SEND = 'send'
    BUS_MIX = 'bus_mix'
    METER = 'meter'
    CUSTOM = 'custom'

@dataclass
class ProcessingNode:
    """Single processing node in the graph."""
    id: str
    node_type: NodeType
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    bypass: bool = False
    _process_fn: Optional[Callable] = field(default=None, repr=False)

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio through this node."""
        if self.bypass or not self.enabled:
            return audio
        if self._process_fn is not None:
            return self._process_fn(audio, self.parameters)
        return self._default_process(audio)

    def _default_process(self, audio: np.ndarray) -> np.ndarray:
        """Default processing based on node type."""
        if self.node_type == NodeType.GAIN:
            gain_db = self.parameters.get('gain_db', 0.0)
            return audio * (10 ** (gain_db / 20.0))

        elif self.node_type == NodeType.GATE:
            threshold_db = self.parameters.get('threshold_db', -40.0)
            threshold_lin = 10 ** (threshold_db / 20.0)
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < threshold_lin:
                range_db = self.parameters.get('range_db', -80.0)
                attenuation = 10 ** (range_db / 20.0)
                return audio * attenuation
            return audio

        elif self.node_type == NodeType.COMPRESSOR:
            threshold_db = self.parameters.get('threshold_db', -20.0)
            ratio = self.parameters.get('ratio', 4.0)
            eps = 1e-10
            level_db = 20 * np.log10(np.abs(audio) + eps)
            over = np.maximum(0, level_db - threshold_db)
            gain_reduction = over * (1 - 1/ratio)
            gain_linear = 10 ** (-gain_reduction / 20.0)
            makeup_db = self.parameters.get('makeup_db', 0.0)
            return audio * gain_linear * (10 ** (makeup_db / 20.0))

        elif self.node_type == NodeType.DELAY:
            delay_samples = int(self.parameters.get('delay_samples', 0))
            if delay_samples > 0 and delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples]
                return delayed
            return audio

        elif self.node_type == NodeType.PAN:
            pan = self.parameters.get('pan', 0.0)
            return audio  # Mono pass-through; stereo handled at bus level

        elif self.node_type in (NodeType.INPUT, NodeType.OUTPUT, NodeType.METER):
            return audio

        return audio

@dataclass
class Connection:
    """Connection between two nodes."""
    source_id: str
    target_id: str
    gain: float = 1.0

class ProcessingGraph:
    """Directed acyclic processing graph."""

    def __init__(self):
        self.nodes: Dict[str, ProcessingNode] = {}
        self.connections: List[Connection] = []
        self._sorted_order: List[str] = []
        self._adjacency: Dict[str, List[str]] = {}
        self._dirty = True

    def add_node(self, node: ProcessingNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self._dirty = True

    def remove_node(self, node_id: str):
        """Remove a node and its connections."""
        self.nodes.pop(node_id, None)
        self.connections = [c for c in self.connections
                          if c.source_id != node_id and c.target_id != node_id]
        self._dirty = True

    def connect(self, source_id: str, target_id: str, gain: float = 1.0):
        """Connect two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Node not found: {source_id} or {target_id}")
        self.connections.append(Connection(source_id, target_id, gain))
        self._dirty = True

    def disconnect(self, source_id: str, target_id: str):
        """Remove connection between two nodes."""
        self.connections = [c for c in self.connections
                          if not (c.source_id == source_id and c.target_id == target_id)]
        self._dirty = True

    def _topological_sort(self):
        """Compute topological ordering of nodes."""
        self._adjacency = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections:
            if conn.source_id in self._adjacency:
                self._adjacency[conn.source_id].append(conn.target_id)
                in_degree[conn.target_id] = in_degree.get(conn.target_id, 0) + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in self._adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            logger.warning("Cycle detected in processing graph!")
            order = list(self.nodes.keys())

        self._sorted_order = order
        self._dirty = False

    def process(self, input_audio: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Process audio through the graph."""
        if self._dirty:
            self._topological_sort()

        buffers: Dict[str, np.ndarray] = {}

        # Initialize input nodes
        for node_id, audio in input_audio.items():
            if node_id in self.nodes:
                buffers[node_id] = audio.copy()

        # Process in topological order
        for node_id in self._sorted_order:
            node = self.nodes[node_id]

            # Collect inputs from connections
            inputs = []
            for conn in self.connections:
                if conn.target_id == node_id and conn.source_id in buffers:
                    inputs.append(buffers[conn.source_id] * conn.gain)

            if inputs:
                # Sum inputs
                max_len = max(len(inp) for inp in inputs)
                combined = np.zeros(max_len)
                for inp in inputs:
                    combined[:len(inp)] += inp
                audio_in = combined
            elif node_id in buffers:
                audio_in = buffers[node_id]
            else:
                continue

            # Process
            buffers[node_id] = node.process(audio_in)

        # Collect outputs
        outputs = {}
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.OUTPUT and node_id in buffers:
                outputs[node_id] = buffers[node_id]

        return outputs if outputs else buffers

    def build_channel_strip(self, channel_id: int) -> List[str]:
        """Build a standard channel strip: Input -> Gate -> EQ -> Comp -> Gain -> Pan -> Output."""
        prefix = f"ch{channel_id}"
        nodes = [
            ProcessingNode(f"{prefix}_input", NodeType.INPUT),
            ProcessingNode(f"{prefix}_gate", NodeType.GATE, {'threshold_db': -40, 'range_db': -80}),
            ProcessingNode(f"{prefix}_eq", NodeType.EQ),
            ProcessingNode(f"{prefix}_comp", NodeType.COMPRESSOR, {'threshold_db': -20, 'ratio': 4.0}),
            ProcessingNode(f"{prefix}_gain", NodeType.GAIN, {'gain_db': 0.0}),
            ProcessingNode(f"{prefix}_pan", NodeType.PAN, {'pan': 0.0}),
            ProcessingNode(f"{prefix}_meter", NodeType.METER),
            ProcessingNode(f"{prefix}_output", NodeType.OUTPUT),
        ]

        for node in nodes:
            self.add_node(node)

        node_ids = [n.id for n in nodes]
        for i in range(len(node_ids) - 1):
            self.connect(node_ids[i], node_ids[i + 1])

        return node_ids

    def get_node(self, node_id: str) -> Optional[ProcessingNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def set_parameter(self, node_id: str, param: str, value: Any):
        """Set a parameter on a node."""
        if node_id in self.nodes:
            self.nodes[node_id].parameters[param] = value

    def get_graph_info(self) -> Dict:
        """Get graph topology info."""
        return {
            'nodes': {nid: {'type': n.node_type.value, 'enabled': n.enabled, 'bypass': n.bypass}
                      for nid, n in self.nodes.items()},
            'connections': [{'from': c.source_id, 'to': c.target_id, 'gain': c.gain}
                           for c in self.connections],
            'order': self._sorted_order if not self._dirty else [],
        }
