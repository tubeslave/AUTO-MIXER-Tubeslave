#!/usr/bin/env python3
"""
Проверка наличия реальных сигналов на BlackHole 64ch.
Читает ~1 сек с устройства, считает RMS по каналам, выводит каналы с сигналом выше порога.
"""
import sys
import time
import numpy as np

# Порог: канал считаем "с сигналом" если RMS > этого уровня (dBFS)
SILENCE_THRESHOLD_DB = -60.0


def main():
    try:
        import pyaudio
    except ImportError:
        print("ERROR: PyAudio not installed")
        sys.exit(1)

    pa = pyaudio.PyAudio()
    device_index = None
    device_name = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "")
        if "BlackHole" in name and "64" in name:
            device_index = i
            device_name = name
            break

    if device_index is None:
        print("BlackHole 64ch not found. Available input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                print(f"  [{i}] {info.get('name')} (inputs: {info.get('maxInputChannels')})")
        pa.terminate()
        sys.exit(1)

    sr = int(pa.get_device_info_by_index(device_index).get("defaultSampleRate", 48000))
    num_ch = 64
    chunk = 4096
    duration_sec = 1.0
    n_chunks = max(1, int(sr * duration_sec / chunk))

    print(f"Device: {device_name} (index={device_index}), {sr} Hz, {num_ch} ch")
    print(f"Recording {duration_sec} s ({n_chunks} chunks)...")

    buf = []
    got_chunks = [0]

    def callback(in_data, frame_count, time_info, status):
        if status:
            print(f"  PyAudio status: {status}")
        buf.append(np.frombuffer(in_data, dtype=np.float32).copy())
        got_chunks[0] += 1
        return (None, pyaudio.paContinue)

    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=num_ch,
        rate=sr,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk,
        stream_callback=callback,
    )
    stream.start_stream()

    while got_chunks[0] < n_chunks and got_chunks[0] < 200:
        time.sleep(0.05)

    stream.stop_stream()
    stream.close()
    pa.terminate()

    if not buf:
        print("No data received.")
        sys.exit(1)

    data = np.concatenate(buf)
    frames = len(data) // num_ch
    data = data[: frames * num_ch].reshape(frames, num_ch)

    # RMS по каналам (float32, 1.0 = 0 dBFS)
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    threshold_linear = 10.0 ** (SILENCE_THRESHOLD_DB / 20.0)
    has_signal = rms > threshold_linear
    channels_with_signal = np.where(has_signal)[0] + 1  # 1-based

    print(f"\nChannels with signal (RMS > {SILENCE_THRESHOLD_DB} dBFS): {len(channels_with_signal)}")
    if len(channels_with_signal) > 0:
        print("  Channels:", list(channels_with_signal))
        print("\nPeak RMS per channel (dBFS), first 24 ch:")
        for ch in range(min(24, num_ch)):
            db = float(rms_db[ch])
            mark = " *" if has_signal[ch] else ""
            print(f"  Ch {ch+1:2d}: {db:6.1f} dB{mark}")
        if num_ch > 24:
            print(f"  ... (ch 25-{num_ch} omitted)")
    else:
        print("  No channels above threshold. RMS (dBFS) first 16 ch:")
        for ch in range(min(16, num_ch)):
            print(f"  Ch {ch+1:2d}: {float(rms_db[ch]):6.1f} dB")

    sys.exit(0 if len(channels_with_signal) > 0 else 1)


if __name__ == "__main__":
    main()
