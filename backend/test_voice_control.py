#!/usr/bin/env python3
"""
Test script for voice control functionality
Tests Faster-Whisper integration and command parsing
"""
import sys
import time
import logging
from voice_control import VoiceControl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test if Whisper model loads correctly"""
    print("\n=== Test 1: Model Loading ===")
    try:
        voice_control = VoiceControl(model_size="small", language="ru")
        print("Loading Whisper model...")
        voice_control.load_model()
        print("✅ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_command_parsing():
    """Test command parsing without audio"""
    print("\n=== Test 2: Command Parsing ===")
    voice_control = VoiceControl()
    
    test_commands = [
        ("канал 1", {"type": "set_fader", "channel": 1}),
        ("канал 5", {"type": "set_fader", "channel": 5}),
        ("гейн 3", {"type": "set_gain", "channel": 3}),
        ("загрузить концерт", {"type": "load_snap", "snap_name": "концерт"}),
        ("снапшот репетиция", {"type": "load_snap", "snap_name": "репетиция"}),
        ("мут 2", {"type": "mute_channel", "channel": 2}),
        ("громче 4", {"type": "volume_up", "channel": 4}),
        ("тише 6", {"type": "volume_down", "channel": 6}),
        ("channel 10", {"type": "set_fader", "channel": 10}),
        ("load test", {"type": "load_snap", "snap_name": "test"}),
    ]
    
    passed = 0
    failed = 0
    
    for text, expected in test_commands:
        result = voice_control._parse_command(text)
        if result:
            # Check if command type matches
            if result.get("type") == expected.get("type"):
                # Check channel if present
                if "channel" in expected:
                    if result.get("channel") == expected.get("channel"):
                        print(f"✅ '{text}' -> {result.get('type')} channel {result.get('channel')}")
                        passed += 1
                    else:
                        print(f"❌ '{text}' -> Expected channel {expected.get('channel')}, got {result.get('channel')}")
                        failed += 1
                # Check snap_name if present
                elif "snap_name" in expected:
                    if result.get("snap_name") == expected.get("snap_name"):
                        print(f"✅ '{text}' -> {result.get('type')} '{result.get('snap_name')}'")
                        passed += 1
                    else:
                        print(f"❌ '{text}' -> Expected snap '{expected.get('snap_name')}', got '{result.get('snap_name')}'")
                        failed += 1
                else:
                    print(f"✅ '{text}' -> {result.get('type')}")
                    passed += 1
            else:
                print(f"❌ '{text}' -> Expected type {expected.get('type')}, got {result.get('type')}")
                failed += 1
        else:
            print(f"❌ '{text}' -> No command recognized")
            failed += 1
    
    print(f"\nParsing results: {passed} passed, {failed} failed")
    return failed == 0


def test_audio_devices():
    """Test if audio devices are available"""
    print("\n=== Test 3: Audio Devices ===")
    try:
        import pyaudio
        audio = pyaudio.PyAudio()
        
        print("Available audio input devices:")
        device_count = audio.get_device_count()
        found_input = False
        
        for i in range(device_count):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} - {info['maxInputChannels']} channels")
                found_input = True
        
        audio.terminate()
        
        if found_input:
            print("✅ Input devices found")
            return True
        else:
            print("⚠️  No input devices found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking audio devices: {e}")
        return False


def test_voice_listening():
    """Test voice listening (requires microphone)"""
    print("\n=== Test 4: Voice Listening ===")
    print("This test will listen for 10 seconds...")
    print("Say a command like 'канал 1' or 'загрузить тест'")
    
    try:
        voice_control = VoiceControl(model_size="small", language="ru")
        voice_control.load_model()
        
        commands_received = []
        
        def on_command(cmd):
            commands_received.append(cmd)
            print(f"\n🎤 Command recognized: {cmd}")
        
        print("\nStarting voice listening...")
        voice_control.start_listening(on_command)
        
        print("Listening for 10 seconds... (speak now!)")
        time.sleep(10)
        
        voice_control.stop_listening()
        
        if commands_received:
            print(f"\n✅ Received {len(commands_received)} command(s)")
            for cmd in commands_received:
                print(f"   - {cmd}")
            return True
        else:
            print("\n⚠️  No commands received. Make sure microphone is working and you spoke clearly.")
            return False
            
    except Exception as e:
        print(f"❌ Error in voice listening test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Voice Control Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Command Parsing", test_command_parsing),
        ("Audio Devices", test_audio_devices),
    ]
    
    # Only run voice listening test if explicitly requested
    if "--listen" in sys.argv:
        tests.append(("Voice Listening", test_voice_listening))
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
