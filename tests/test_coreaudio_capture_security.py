"""Static regression checks for the macOS-only CoreAudio render boundary."""

from pathlib import Path


def test_render_callback_rejects_frames_beyond_allocated_capacity_before_byte_size_write():
    source = Path("backend/native/src/audio/coreaudio_capture.cpp").read_text(encoding="utf-8")
    callback = source[source.index("OSStatus CoreAudioCapture::renderCallback(") :]

    capacity_guard = callback.index("inNumberFrames > self->allocated_frame_capacity_")
    byte_size_write = callback.index("mDataByteSize = inNumberFrames * sizeof(float)")

    assert capacity_guard < byte_size_write
    assert "return kAudio_ParamError;" in callback[capacity_guard:byte_size_write]


def test_allocation_and_callback_share_the_same_frame_capacity_member():
    source = Path("backend/native/src/audio/coreaudio_capture.cpp").read_text(encoding="utf-8")

    assert "allocated_frame_capacity_ = static_cast<UInt32>(block_size_) * 2U" in source
    assert "calloc(allocated_frame_capacity_, sizeof(float))" in source
    assert "inNumberFrames > self->allocated_frame_capacity_" in source
