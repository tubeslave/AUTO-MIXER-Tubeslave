#!/usr/bin/env python3
"""
Ableton Parameter Discovery — логирование OSC от Ableton для маппинга параметров.

Запуск:
  cd backend && PYTHONPATH=. python ../scripts/ableton_param_discovery.py

Скрипт:
- подключается к AbletonOSC
- запрашивает названия параметров устройств
- включает listeners для параметров mixer device на track 0
- печатает изменения параметров в реальном времени
"""
import sys
import os
import time
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ableton_client import AbletonClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    print("\n=== Ableton Parameter Discovery ===\n")
    print("Подключаюсь к Ableton (127.0.0.1:11000)...")
    client = AbletonClient(
        ip="127.0.0.1",
        send_port=11000,
        recv_port=11001,
        channel_offset=0,
        utility_device_index=0,
        eq_eight_device_index=1,
    )
    if not client.connect():
        print("Ошибка: не удалось подключиться к Ableton.")
        print("Убедитесь, что Ableton Live запущен и AbletonOSC активирован.")
        return 1

    print("Подключено.\n")

    # Перехватываем ответы и выводим параметры
    param_responses = {}
    mixer_param_names = []

    def capture_params(address, *args):
        if address == '/live/device/get/parameters/name' and len(args) >= 3:
            track_id = int(args[0])
            device_id = int(args[1])
            names = [str(a) for a in args[2:]]
            key = (track_id, device_id)
            param_responses[key] = names
            print(f"\n--- Device track={track_id} device_id={device_id} ---")
            for i, n in enumerate(names):
                marker = "  <-- нужен для phase" if "inv" in n.lower() or "phase" in n.lower() else ""
                print(f"  [{i}] {n}{marker}")
            print()
            if track_id == 0 and device_id == 0:
                mixer_param_names.clear()
                mixer_param_names.extend(names)

        if address == '/live/device/get/parameter/value' and len(args) >= 4:
            track_id = int(args[0])
            device_id = int(args[1])
            param_id = int(args[2])
            value = args[3]
            param_name = None
            if track_id == 0 and device_id == 0 and param_id < len(mixer_param_names):
                param_name = mixer_param_names[param_id]
            label = f" [{param_name}]" if param_name else ""
            print(
                f">>> CHANGE track={track_id} device={device_id} "
                f"param={param_id}{label} value={value}"
            )

    client.subscribe("*", capture_params)

    # Запрос параметров для треков 0..2, устройств 0..4 (Utility + EQ Eight + запас)
    print("Запрашиваю параметры (track 0..2, device 0..4)...\n")
    for track_id in range(3):
        for dev_id in range(5):
            client._send_osc('/live/device/get/parameters/name', track_id, dev_id)
            time.sleep(0.12)

    time.sleep(1.5)

    # Включаем listeners для mixer device первого трека, чтобы ловить Track Delay
    if mixer_param_names:
        print("Включаю listeners для mixer device на track 0...\n")
        for param_id, param_name in enumerate(mixer_param_names):
            client._send_osc('/live/device/start_listen/parameter/value', 0, 0, param_id)
            print(f"  listen: track=0 device=0 param={param_id} name={param_name}")
            time.sleep(0.05)
        print()

    print("\n" + "=" * 60)
    print("Результаты выше. Сейчас скрипт слушает mixer device track 0.")
    print("Для начала меняйте Track Delay на первом треке Ableton.")
    print("Нажмите Ctrl+C для выхода.")
    print("=" * 60)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nВыход.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
