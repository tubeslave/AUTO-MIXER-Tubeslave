from __future__ import annotations

import queue
import threading
import time
from typing import Any

from pythonosc import dispatcher, osc_server, udp_client

class AbletonOSC:
    """Small request/reply client for the AbletonOSC Remote Script."""
    def __init__(self, host: str="127.0.0.1", send_port: int=11000, receive_port: int=11001,
                 timeout_s: float=1.5):
        self.client=udp_client.SimpleUDPClient(host,send_port)
        self.timeout_s=timeout_s
        self.q: queue.Queue=queue.Queue()
        d=dispatcher.Dispatcher(); d.set_default_handler(self._reply)
        self.server=osc_server.ThreadingOSCUDPServer((host,receive_port),d)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def _reply(self,address,*args):
        self.q.put((address,args))

    def send(self,address: str,args: list[Any] | None=None):
        self.client.send_message(address,args or [])

    def query(self,address: str,args: list[Any] | None=None, expected: str | None=None) -> dict[str,Any]:
        while not self.q.empty():
            try:self.q.get_nowait()
            except queue.Empty:break
        self.send(address,args)
        deadline=time.time()+self.timeout_s
        expected=expected or address.replace("/get/","/get/")
        while time.time()<deadline:
            try:
                a,v=self.q.get(timeout=max(.01,deadline-time.time()))
                if a==expected:
                    return {"address":a,"args":list(v)}
            except queue.Empty:
                break
        raise TimeoutError(f"No AbletonOSC response for {address}")

    def close(self):
        self.server.shutdown(); self.server.server_close()

def set_track_volume(track_id: int, value: float, host="127.0.0.1", send_port=11000):
    c=udp_client.SimpleUDPClient(host,send_port)
    c.send_message("/live/track/set/volume",[track_id,float(value)])
    return {"sent":True,"track_id":track_id,"value":float(value)}

def set_device_parameter(track_id: int, device_id: int, parameter_id: int, value: float,
                         host="127.0.0.1", send_port=11000):
    c=udp_client.SimpleUDPClient(host,send_port)
    c.send_message("/live/device/set/parameter/value",
                   [track_id,device_id,parameter_id,float(value)])
    return {"sent":True,"track_id":track_id,"device_id":device_id,
            "parameter_id":parameter_id,"value":float(value),
            "warning":"write must be followed by readback/render verification before acceptance"}
