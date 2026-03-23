#!/usr/bin/env python3
"""
AUTO-MIXER Tubeslave — единый запуск backend + frontend.
"""

import os
import signal
import subprocess
import sys
import time

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def print_status(message: str, status: str = "info"):
    colors = {"info": "", "success": GREEN, "warning": YELLOW, "error": RED}
    print(f"{colors.get(status, '')}{message}{RESET}")


def start_backend():
    """Запустить backend (WebSocket server)."""
    print_status("\n🚀 Запуск backend...", "info")
    root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root, "backend")
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def start_frontend():
    """Запустить frontend (React dev server)."""
    print_status("\n🌐 Запуск frontend...", "info")
    root = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root, "frontend")
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print_status("  📦 Установка npm зависимостей...", "warning")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
    proc = subprocess.Popen(
        ["npm", "start"],
        cwd=frontend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║   🎛️  AUTO-MIXER Tubeslave — Auto Mixer                        ║
║   Backend + Frontend                                           ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    backend_proc = start_backend()
    try:
        frontend_proc = start_frontend()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_status("  ⚠️ npm не найден — пропускаем frontend", "warning")
        frontend_proc = None

    print("\n" + "=" * 60)
    print_status("✅ AUTO-MIXER запущен!", "success")
    print("=" * 60)
    print("""
    🌐 Frontend:   http://localhost:3000
    🔌 WebSocket:  ws://localhost:8765

    Нажмите Ctrl+C для остановки
    """)

    try:
        while True:
            time.sleep(3)
            if backend_proc.poll() is not None:
                print_status("❌ Backend остановлен", "error")
                break
            if frontend_proc and frontend_proc.poll() is not None:
                print_status("❌ Frontend остановлен", "error")
                break
    except KeyboardInterrupt:
        pass

    print_status("\n🛑 Остановка...", "warning")
    backend_proc.terminate()
    if frontend_proc:
        frontend_proc.terminate()
    print_status("✅ Остановлено", "success")


if __name__ == "__main__":
    main()
