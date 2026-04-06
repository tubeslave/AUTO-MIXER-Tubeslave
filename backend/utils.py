"""Utility functions for AutoMixer backend."""

import logging
import os
from typing import Any

import numpy as np


def setup_file_logging() -> str:
    """
    Дублировать INFO в logs/automixer-backend.log (удобно смотреть EQ после анализа).
    Returns the log file path.
    """
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "automixer-backend.log")
    abs_path = os.path.abspath(log_path)
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(getattr(h, "baseFilename", "")) == abs_path:
                    return log_path
            except (OSError, TypeError, ValueError):
                pass
    fmt = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    # Ensure root logger level allows INFO messages (blake2 errors may have
    # already called basicConfig, leaving root at WARNING).
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(fh)
    return log_path


def convert_numpy_types(obj: Any) -> Any:
    """
    Рекурсивно преобразует NumPy типы в нативные Python типы для JSON сериализации.

    Args:
        obj: Объект, который может содержать NumPy типы

    Returns:
        Объект с преобразованными типами
    """
    # Проверяем на NumPy скалярные типы
    if isinstance(obj, np.generic):
        if isinstance(
            obj,
            (
                np.integer,
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            # Для других NumPy типов пытаемся преобразовать в Python тип
            try:
                return obj.item()
            except (AttributeError, ValueError):
                return (
                    float(obj)
                    if np.issubdtype(type(obj), np.floating)
                    else int(obj)
                    if np.issubdtype(type(obj), np.integer)
                    else bool(obj)
                )
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
