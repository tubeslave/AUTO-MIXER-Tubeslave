#!/usr/bin/env python3
"""
Simple interactive test for voice recognition
Tests voice recognition with microphone input
"""
import sys
import time
import logging
from voice_control import VoiceControl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("Simple Voice Recognition Test")
    print("=" * 60)
    print("\nThis test will:")
    print("1. Load the Whisper model")
    print("2. Start listening to your microphone")
    print("3. Recognize and parse voice commands")
    print("\nTry saying commands like:")
    print("  - 'канал 1'")
    print("  - 'загрузить тест'")
    print("  - 'гейн 5'")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Initialize voice control
        print("Initializing voice control...")
        voice_control = VoiceControl(model_size="small", language="ru")
        
        print("Loading Whisper model (this may take a moment)...")
        voice_control.load_model()
        print("✅ Model loaded!")
        
        commands_received = []
        
        def on_command(cmd):
            commands_received.append(cmd)
            print(f"\n🎤 Command recognized: {cmd}")
            print(f"   Type: {cmd.get('type')}")
            if 'channel' in cmd:
                print(f"   Channel: {cmd.get('channel')}")
            if 'snap_name' in cmd:
                print(f"   Snapshot: {cmd.get('snap_name')}")
        
        print("\nStarting voice listening...")
        print("Speak now! (Listening for 30 seconds or until Ctrl+C)\n")
        
        voice_control.start_listening(on_command)
        
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        voice_control.stop_listening()
        
        print(f"\n✅ Test completed!")
        print(f"   Commands received: {len(commands_received)}")
        
        if commands_received:
            print("\nRecognized commands:")
            for i, cmd in enumerate(commands_received, 1):
                print(f"   {i}. {cmd}")
        else:
            print("\n⚠️  No commands were recognized.")
            print("   Make sure:")
            print("   - Microphone is working")
            print("   - You spoke clearly")
            print("   - You used supported commands")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
