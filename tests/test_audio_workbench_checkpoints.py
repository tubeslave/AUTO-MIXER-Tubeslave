from audio_workbench.checkpoints import commit, valid, pending

def test_checkpoint_invalidates_on_bytes_or_params(tmp_path):
    p=tmp_path/"x.bin";p.write_bytes(b"one")
    commit(str(tmp_path),"drums",str(p),{"src":"abc"},{"v":1})
    assert valid(str(tmp_path),"drums",{"src":"abc"},{"v":1})["valid"]
    assert not valid(str(tmp_path),"drums",{"src":"abc"},{"v":2})["valid"]
    p.write_bytes(b"two")
    assert not valid(str(tmp_path),"drums",{"src":"abc"},{"v":1})["valid"]

def test_pending_only_returns_invalid_nodes(tmp_path):
    p=tmp_path/"x";p.write_bytes(b"x")
    commit(str(tmp_path),"bass",str(p),{"src":"1"},{})
    g=[{"key":"bass","inputs":{"src":"1"},"params":{}},{"key":"vocals","inputs":{"src":"2"},"params":{}}]
    assert [x["key"] for x in pending(str(tmp_path),g)]==["vocals"]
