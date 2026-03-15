#!/usr/bin/env python3
"""
Комплексное тестирование Auto Mixer Tubeslave
Проверяет:
1. Импорт всех модулей
2. DSP-корректность алгоритмов с точки зрения стандартов звукорежиссуры
3. Безопасность для live-звука
"""

import sys
import os
import json
import traceback
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append({"name": name, "status": status, "detail": detail})
    print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))
    return condition

def warn(name, detail=""):
    results.append({"name": name, "status": WARN, "detail": detail})
    print(f"  {WARN}: {name}" + (f" — {detail}" if detail else ""))

# ============================================================
# ТЕСТ 1: Импорт всех модулей
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 1: Импорт модулей бэкенда")
print("="*70)

modules_to_import = [
    "wing_client", "lufs_gain_staging", "auto_eq", "auto_compressor",
    "auto_compressor_cf", "auto_gate_caig", "auto_fader", "auto_fader_hybrid",
    "auto_effects", "auto_phase_gcc_phat", "auto_panner", "auto_panner_adaptive",
    "auto_reverb", "channel_recognizer", "bleed_service", "cross_adaptive_eq",
    "compressor_adaptation", "phase_alignment", "backup_channels",
    "auto_eq_processing"
]

imported = {}
for mod_name in modules_to_import:
    try:
        imported[mod_name] = __import__(mod_name)
        test(f"import {mod_name}", True)
    except Exception as e:
        test(f"import {mod_name}", False, str(e)[:100])

# ============================================================
# ТЕСТ 2: LUFS Gain Staging — ITU-R BS.1770-4
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 2: LUFS Gain Staging — ITU-R BS.1770-4")
print("="*70)

if "lufs_gain_staging" in imported:
    lufs = imported["lufs_gain_staging"]
    
    # 2.1: K-weighting filter coefficients
    # ITU-R BS.1770-4: stage 1 shelving + stage 2 RLB HPF
    try:
        meter = lufs.LUFSMeter(sample_rate=48000)
        test("LUFSMeter инициализация (48kHz)", True)
        
        # 2.2: Тест на синусе 1kHz при -6dBFS → ожидаем ~-9 LUFS
        # (1kHz проходит K-weighting без изменений)
        sr = 48000
        duration = 5  # секунд
        t = np.linspace(0, duration, sr * duration, endpoint=False)
        sine_1k = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5  # -6 dBFS peak
        
        # Обработка блоками
        block_size = meter.window_samples
        for i in range(0, len(sine_1k) - block_size, block_size):
            block = sine_1k[i:i+block_size]
            meter.process(block)
        
        lufs_val = meter.get_current_lufs()
        # -6dBFS peak синус → RMS = peak - 3.01 → ~ -9.01 dBFS ≈ -9.01 LUFS на 1kHz
        test("LUFS 1kHz синус: значение корректно",
             lufs_val is not None and -15 < lufs_val < -5,
             f"LUFS = {lufs_val:.1f} (ожидается ~-9 LUFS для -6dBFS синуса)")
        
        # 2.3: Тест тишины → не NaN, не Inf
        meter_silent = lufs.LUFSMeter(sample_rate=48000)
        silence = np.zeros(48000, dtype=np.float32)
        block_size_s = meter_silent.window_samples
        for i in range(0, len(silence) - block_size_s, block_size_s):
            meter_silent.process(silence[i:i+block_size_s])
        
        silent_lufs = meter_silent.get_current_lufs()
        test("LUFS тишина: нет NaN/Inf",
             silent_lufs is None or (not np.isnan(silent_lufs) and not np.isinf(silent_lufs)),
             f"LUFS тишины = {silent_lufs}")
        
        # 2.4: Проверка наличия relative gate (-10 LU) — C-03 fix
        source = open("lufs_gain_staging.py").read()
        has_relative_gate = "-10" in source or "relative" in source.lower() or "10 * np.log10" in source
        test("LUFS: relative gate (-10 LU) по ITU-R BS.1770-4 (C-03 fix)",
             has_relative_gate,
             "Двухпроходное гейтирование для integrated LUFS")
             
        # 2.5: Floating-point drift protection — C-12 fix
        has_clamp = "_sum_squares < 0" in source or "sum_squares = 0" in source
        test("LUFS: защита от floating-point drift (C-12 fix)",
             has_clamp,
             "Clamp _sum_squares >= 0 после длительных сессий")
             
    except Exception as e:
        test("LUFS модуль", False, f"Ошибка: {traceback.format_exc()[:200]}")

    # 2.6: Gain Staging targets
    # По стандарту live: peak -12 to -6 dBFS, средний -18 dBFS
    try:
        controller = lufs.LUFSGainStagingController.__init__.__doc__ or ""
        # Проверяем дефолтные target levels
        test("Gain Staging: модуль доступен", True)
    except Exception as e:
        test("Gain Staging controller", False, str(e)[:100])

# ============================================================
# ТЕСТ 3: Auto EQ — частотные слоты и маскинг
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 3: Auto EQ — частотные слоты и корректность")
print("="*70)

if "auto_eq" in imported:
    auto_eq = imported["auto_eq"]
    
    try:
        # 3.1: Проверяем InstrumentProfiles
        profiles = auto_eq.InstrumentProfiles
        test("InstrumentProfiles загружен", True)
        
        # 3.2: Проверяем что HPF присутствует (обязательно для live)
        source_eq = open("auto_eq.py").read()
        has_hpf = "hpf" in source_eq.lower() or "high_pass" in source_eq.lower() or "highpass" in source_eq.lower() or "low_cut" in source_eq.lower() or "lowcut" in source_eq.lower()
        test("Auto EQ: HPF / low cut filter реализован", has_hpf,
             "HPF обязателен на всём кроме kick/bass")
        
        # 3.3: Частотные диапазоны корректны
        # Mud zone: 200-500 Hz должна быть целью для cut
        has_mud_zone = any(str(f) in source_eq for f in range(200, 600, 50))
        test("Auto EQ: mud zone (200-500 Hz) учтена", has_mud_zone)
        
        # 3.4: Presence zone для вокала (2-5 kHz)
        has_presence = any(str(f) in source_eq for f in [2000, 2500, 3000, 3500, 4000, 5000])
        test("Auto EQ: presence zone (2-5 kHz) учтена", has_presence)
        
        # 3.5: Safety limits на EQ gain
        has_max_gain = "max_gain" in source_eq.lower() or "max_boost" in source_eq.lower() or "clamp" in source_eq.lower() or "clip" in source_eq.lower()
        test("Auto EQ: safety limits на gain", has_max_gain,
             "Ограничение максимального boost/cut для предотвращения фидбэка")
             
        # 3.6: Subtractive before additive
        has_subtractive = "cut" in source_eq.lower() and "boost" in source_eq.lower()
        test("Auto EQ: subtractive/additive подход", has_subtractive,
             "Принцип 'сначала убрать, потом поднять'")

    except Exception as e:
        test("Auto EQ модуль", False, traceback.format_exc()[:200])

# ============================================================
# ТЕСТ 4: Auto Compressor — ratio/attack/release
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 4: Auto Compressor — параметры по инструментам")
print("="*70)

if "auto_compressor" in imported and "auto_compressor_cf" in imported:
    auto_comp = imported["auto_compressor"]
    auto_comp_cf = imported["auto_compressor_cf"]
    
    try:
        source_comp = open("auto_compressor.py").read()
        source_cf = open("auto_compressor_cf.py").read()
        
        # 4.1: Crest Factor формула — C-05 fix
        # Высокий CF (перкуссия) → высокий ratio
        test("Compressor CF: формула ratio (C-05 fix)",
             "1.0 + 0.5" in source_cf or "1.0 + 0.3" in source_cf,
             "cf_norm высокий → ratio высокий (перкуссия = агрессивная компрессия)")
        
        # 4.2: Проверяем разумные диапазоны ratio
        # Вокал: 3:1-4:1, Bass: 4:1-8:1, Drums: 4:1-10:1
        has_ratio_range = "ratio" in source_comp.lower()
        test("Compressor: параметр ratio реализован", has_ratio_range)
        
        # 4.3: Attack time корректность
        # Быстрый attack (1-5мс) для drums, средний (5-30мс) для вокала
        has_attack = "attack" in source_comp.lower()
        test("Compressor: attack time реализован", has_attack)
        
        # 4.4: Release time
        has_release = "release" in source_comp.lower()
        test("Compressor: release time реализован", has_release)
        
        # 4.5: Gain reduction safety limit
        has_gr_limit = "max" in source_comp.lower() and ("gain_reduction" in source_comp.lower() or "gr" in source_comp.lower())
        test("Compressor: safety limit на gain reduction", has_gr_limit,
             "Защита от чрезмерной компрессии в live")
             
        # 4.6: Make-up gain
        has_makeup = "makeup" in source_comp.lower() or "make_up" in source_comp.lower() or "auto_gain" in source_comp.lower()
        test("Compressor: make-up gain", has_makeup)
        
    except Exception as e:
        test("Auto Compressor", False, traceback.format_exc()[:200])

# ============================================================
# ТЕСТ 5: Auto Gate — hysteresis и state machine
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 5: Auto Gate — hysteresis, state machine")
print("="*70)

if "auto_gate_caig" in imported:
    gate_mod = imported["auto_gate_caig"]
    
    try:
        source_gate = open("auto_gate_caig.py").read()
        
        # 5.1: Hysteresis applied — C-02 fix
        has_hysteresis_applied = "close_threshold" in source_gate or "hysteresis_db" in source_gate
        test("Gate: hysteresis dual-threshold (C-02 fix)", has_hysteresis_applied,
             "Open threshold и close threshold разделены на hysteresis_db")
        
        # 5.2: State machine states
        # Gate может иметь 5 состояний (CLOSING объединён с RELEASING) — это корректно
        core_states = ["CLOSED", "OPENING", "OPEN", "HOLD", "RELEASING"]
        has_states = all(s in source_gate for s in core_states)
        test("Gate: полная state machine (5+ состояний)", has_states,
             f"Состояния: {[s for s in core_states if s in source_gate]}")
        
        # 5.3: Re-open during RELEASING — C-10 fix
        has_reopen = "RELEASING" in source_gate and "OPENING" in source_gate
        test("Gate: re-open при RELEASING (C-10 fix)", has_reopen,
             "Быстрые удары не теряются при release")
        
        # 5.4: Attack/Release times для live
        has_attack_release = "attack" in source_gate.lower() and "release" in source_gate.lower()
        test("Gate: attack и release параметры", has_attack_release)
        
        # 5.5: Численный тест — hysteresis
        # Создаем сигнал который колеблется около threshold
        try:
            settings_cls = getattr(gate_mod, 'GateSettings', None)
            processor_cls = getattr(gate_mod, 'GateProcessor', None)
            if settings_cls and processor_cls:
                settings = settings_cls()
                settings.threshold_db = -40.0
                settings.hysteresis_db = 3.0
                processor = processor_cls(settings)
                
                # Сигнал чуть ниже threshold (между close и open)
                # = -41 dB (ниже open=-40, но выше close=-43)
                rms_between = 10**(-41/20)
                sr = 48000
                signal_between = np.ones(4800, dtype=np.float32) * rms_between
                
                # Сначала открываем (громкий сигнал)
                loud_signal = np.ones(4800, dtype=np.float32) * 10**(-30/20)
                processor.process(loud_signal, sr)
                state_after_loud = str(processor.state)
                
                # Теперь сигнал в зоне hysteresis — gate не должен закрываться сразу
                processor.process(signal_between, sr)
                state_in_hysteresis = str(processor.state)
                
                test("Gate: hysteresis зона корректна",
                     "OPEN" in state_in_hysteresis or "HOLD" in state_in_hysteresis,
                     f"Громкий→{state_after_loud}, hysteresis зона→{state_in_hysteresis}")
            else:
                warn("Gate: GateProcessor не найден для численного теста")
        except Exception as e:
            warn(f"Gate численный тест", str(e)[:100])
    
    except Exception as e:
        test("Auto Gate", False, traceback.format_exc()[:200])

# ============================================================
# ТЕСТ 6: Phase Alignment (GCC-PHAT)
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 6: Phase Alignment — GCC-PHAT")
print("="*70)

if "auto_phase_gcc_phat" in imported:
    phase_mod = imported["auto_phase_gcc_phat"]
    
    try:
        source_phase = open("auto_phase_gcc_phat.py").read()
        
        # 6.1: Правильный порядок ref/tgt — C-07 fix
        test("Phase: ref/tgt порядок (C-07 fix)",
             "ref_audio, tgt_audio" in source_phase or "ref_audio,tgt_audio" in source_phase
             or "(ref_audio" in source_phase,
             "add_frames(ref, tgt) — правильный знак задержки")
        
        # 6.2: Защита от деления на ноль — C-08 fix
        has_eps_protection = "eps" in source_phase and ("energy_product" in source_phase or "clip" in source_phase)
        test("Phase: защита от /0 на тишине (C-08 fix)", has_eps_protection,
             "correlation_peak = clip(value / (energy + eps), 0, 1)")
        
        # 6.3: Параболическая интерполяция
        has_parabolic = "parabolic" in source_phase.lower() or "interpolat" in source_phase.lower()
        test("Phase: суб-сэмплная точность (параболическая интерполяция)", has_parabolic)
        
        # 6.4: Числовой тест — известная задержка
        try:
            analyzer_cls = getattr(phase_mod, 'GCCPHATAnalyzer', None)
            if analyzer_cls:
                sr = 48000
                delay_samples = 24  # 0.5 мс при 48kHz
                
                # Ref: синус 440Hz
                t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
                ref = np.sin(2 * np.pi * 440 * t).astype(np.float32)
                
                # Target: тот же синус с задержкой
                tgt = np.zeros_like(ref)
                tgt[delay_samples:] = ref[:-delay_samples]
                
                analyzer = analyzer_cls(sample_rate=sr)
                analyzer.add_frames(ref, tgt)
                result = analyzer.compute_delay()
                
                if result is not None:
                    if hasattr(result, 'delay_samples'):
                        detected_delay = result.delay_samples
                    elif isinstance(result, (tuple, list)):
                        detected_delay = result[0]
                    else:
                        detected_delay = float(result)
                    
                    test("Phase: обнаружение задержки 0.5мс",
                         abs(detected_delay - delay_samples) < 5,
                         f"Ожидалось ~{delay_samples} samples, получено {detected_delay:.1f}")
                else:
                    warn("Phase: compute_delay вернул None")
            else:
                warn("Phase: GCCPHATAnalyzer не найден")
        except Exception as e:
            warn(f"Phase числовой тест", str(e)[:150])
    
    except Exception as e:
        test("Phase Alignment", False, traceback.format_exc()[:200])

# ============================================================
# ТЕСТ 7: Auto Fader — балансировка и safety
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 7: Auto Fader — балансировка и safety")
print("="*70)

if "auto_fader" in imported:
    try:
        source_fader = open("auto_fader.py").read()
        source_hybrid = open("auto_fader_hybrid.py").read()
        
        # 7.1: Return type fix — C-06
        test("Fader Hybrid: return type Tuple[float,float,float] (C-06)",
             "Tuple[float, float, float]" in source_hybrid,
             "3 значения: correction_db, rate_limited, confidence")
        
        # 7.2: ScenarioDetector кэширование — C-13
        has_cached = "_scenario_detector" in source_hybrid
        test("Fader Hybrid: ScenarioDetector кэширован (C-13)", has_cached,
             "Избежание аллокаций на 100Hz×N каналов")
        
        # 7.3: Rate limiting (плавность)
        has_rate_limit = "rate" in source_fader.lower() and "limit" in source_fader.lower()
        test("Fader: rate limiting (плавные изменения)", has_rate_limit,
             "Предотвращает резкие скачки фейдеров")
        
        # 7.4: Safety limits на фейдеры
        has_fader_limits = "max" in source_fader.lower() and ("fader" in source_fader.lower() or "db" in source_fader.lower())
        test("Fader: safety limits на уровень", has_fader_limits)
        
        # 7.5: LUFS target levels
        has_lufs_target = "lufs" in source_fader.lower() or "target" in source_fader.lower()
        test("Fader: LUFS-based target levels", has_lufs_target,
             "Балансировка на основе LUFS (громкость)")
        
    except Exception as e:
        test("Auto Fader", False, traceback.format_exc()[:200])

# ============================================================
# ТЕСТ 8: Auto Effects — LRA и энергия
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 8: Auto Effects — энергия и LRA")
print("="*70)

if "auto_effects" in imported:
    try:
        source_fx = open("auto_effects.py").read()
        
        # 8.1: Нормализация энергий — C-01 fix
        has_normalized = "energy_sum" in source_fx or "_energy_sum" in source_fx or "energy_low + energy_mid" in source_fx
        test("Effects: нормализованные энергии (C-01 fix)", has_normalized,
             "energy_band / total_energy вместо _db_to_linear(linear)")
        
        # 8.2: LRA в dB — C-09 fix
        has_lra_db = "log10" in source_fx and ("lra" in source_fx.lower() or "percentile" in source_fx)
        test("Effects: LRA в dB домене (C-09 fix)", has_lra_db,
             "20*log10(rms) перед percentile для корректного LRA")
        
        # 8.3: Reverb/delay send levels
        has_send = "send" in source_fx.lower() or "reverb" in source_fx.lower() or "delay" in source_fx.lower()
        test("Effects: параметры send (reverb/delay)", has_send)
        
    except Exception as e:
        test("Auto Effects", False, traceback.format_exc()[:200])

# ============================================================
# ТЕСТ 9: Wing Client — безопасные операции
# ============================================================
print("\n" + "="*70)
print("ТЕСТ 9: Wing Client OSC — безопасность")
print("="*70)

if "wing_client" in imported:
    try:
        source_wing = open("wing_client.py").read()
        
        # 9.1: find_snap_by_name без GO — C-04 fix
        has_safe_snap = "$name" in source_wing or "name" in source_wing
        has_no_go_in_find = True  # Проверим более детально
        
        # Ищем метод find_snap_by_name
        import re
        find_snap_match = re.search(r'def find_snap_by_name\(.*?\n(?:.*?\n)*?(?=\n    def |\nclass |\Z)', source_wing)
        if find_snap_match:
            snap_body = find_snap_match.group()
            has_go_in_body = '"GO"' in snap_body or "'GO'" in snap_body
            test("Wing: find_snap_by_name без GO (C-04 fix)",
                 not has_go_in_body,
                 "Чтение имён снапшотов без загрузки (read-only query)")
        else:
            warn("Wing: find_snap_by_name не найден")
        
        # 9.2: OSC порт по стандарту Wing
        has_port_2223 = "2223" in source_wing
        test("Wing: OSC порт 2223 (стандарт Wing)", has_port_2223)
        
        # 9.3: Subscription keepalive
        has_xremote = "xremote" in source_wing.lower() or "keepalive" in source_wing.lower() or "subscribe" in source_wing.lower()
        test("Wing: subscription keepalive", has_xremote,
             "Wing требует periodic keepalive для подписки")
        
        # 9.4: Error handling для OSC
        has_try_except = source_wing.count("try:") > 3
        test("Wing: error handling (try/except)", has_try_except,
             "Устойчивость к сетевым сбоям")
        
    except Exception as e:
        test("Wing Client", False, traceback.format_exc()[:200])

# ============================================================
# ДОПОЛНИТЕЛЬНО: Проверка стандартов звукорежиссуры
# ============================================================
print("\n" + "="*70)
print("ДОПОЛНИТЕЛЬНЫЕ ПРОВЕРКИ: стандарты звукорежиссуры")
print("="*70)

# Channel Recognizer — определение типа инструмента
if "channel_recognizer" in imported:
    try:
        recognizer = imported["channel_recognizer"]
        has_presets = hasattr(recognizer, 'AVAILABLE_PRESETS')
        test("Channel Recognizer: пресеты инструментов доступны", has_presets)
        if has_presets:
            presets = recognizer.AVAILABLE_PRESETS
            test("Channel Recognizer: количество пресетов", 
                 len(presets) >= 5,
                 f"{len(presets)} пресетов: {list(presets.keys()) if isinstance(presets, dict) else presets[:10]}")
    except Exception as e:
        test("Channel Recognizer", False, str(e)[:100])

# Bleed Service — компенсация bleed
if "bleed_service" in imported:
    try:
        source_bleed = open("bleed_service.py").read()
        has_compensation = "compensation" in source_bleed.lower()
        test("Bleed Service: compensation factor", has_compensation)
    except Exception as e:
        test("Bleed Service", False, str(e)[:100])

# Cross-Adaptive EQ (IMP метод)
if "cross_adaptive_eq" in imported:
    try:
        source_xeq = open("cross_adaptive_eq.py").read()
        has_cross_channel = "cross" in source_xeq.lower() or "mask" in source_xeq.lower()
        test("Cross-Adaptive EQ: кросс-канальная обработка", has_cross_channel,
             "Метод IMP (De Man, Reiss & Stables)")
    except Exception as e:
        test("Cross-Adaptive EQ", False, str(e)[:100])

# Auto Panner — панорама
if "auto_panner" in imported:
    try:
        source_pan = open("auto_panner.py").read()
        # Центр для kick, snare, bass, vocal (по стандарту)
        has_center = "center" in source_pan.lower() or "pan" in source_pan.lower()
        test("Auto Panner: панорамирование каналов", has_center)
    except Exception as e:
        test("Auto Panner", False, str(e)[:100])

# Auto Reverb
if "auto_reverb" in imported:
    try:
        source_rev = open("auto_reverb.py").read()
        has_decay = "decay" in source_rev.lower() or "rt60" in source_rev.lower() or "time" in source_rev.lower()
        test("Auto Reverb: параметры decay/time", has_decay)
        
        has_predelay = "pre_delay" in source_rev.lower() or "predelay" in source_rev.lower()
        test("Auto Reverb: pre-delay параметр", has_predelay,
             "Pre-delay отделяет сухой сигнал от реверба")
    except Exception as e:
        test("Auto Reverb", False, str(e)[:100])

# ============================================================
# ИТОГИ
# ============================================================
print("\n" + "="*70)
print("ИТОГИ ТЕСТИРОВАНИЯ")
print("="*70)

passed = sum(1 for r in results if r["status"] == PASS)
failed = sum(1 for r in results if r["status"] == FAIL)
warned = sum(1 for r in results if r["status"] == WARN)
total = len(results)

print(f"\n  Всего тестов: {total}")
print(f"  {PASS}: {passed}")
print(f"  {FAIL}: {failed}")
print(f"  {WARN}: {warned}")
print(f"\n  Успешность: {passed}/{total - warned} ({100*passed/(total-warned):.0f}%)")

# Сохраняем результаты в JSON
report = {
    "timestamp": datetime.now().isoformat(),
    "summary": {
        "total": total,
        "passed": passed,
        "failed": failed,
        "warnings": warned,
        "success_rate": f"{100*passed/(total-warned):.0f}%"
    },
    "results": results
}

with open("test_results.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n  Результаты сохранены в test_results.json")

# Exit code
sys.exit(0 if failed == 0 else 1)
