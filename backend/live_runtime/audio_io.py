from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class AudioDevice:
    id:str
    name:str
    backend:str
    inputs:int
    outputs:int
    sample_rates:tuple[int,...]=()

class AudioInputAdapter(Protocol):
    def list_devices(self)->list[AudioDevice]: ...
    def open(self,device_id:str,sample_rate:int,channels:list[int])->None: ...
    def read(self,frames:int): ...
    def close(self)->None: ...

class RoutingAdapter(Protocol):
    def inspect(self)->dict: ...
    def verify(self,expected:dict)->dict: ...
    def apply_preset(self,name:str)->dict: ...

# Concrete adapters:
# - WingUSBAdapter: ASIO/CoreAudio 48x48
# - DanteAudioAdapter: DVS / Dante PCIe / supported host audio device
# - DanteManagedRoutingAdapter: optional, only when an authorized API is available
