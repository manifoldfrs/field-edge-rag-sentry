#!/usr/bin/env python3
"""
ring_buffer.py
--------------
A minimal, thread-safe fixed-capacity ring buffer.

Intended for FPS / latency tracking where we only care about the N most-recent
samples.

Usage
-----
>>> rb = RingBuffer(100)
>>> rb.append(42)
>>> rb.mean()
42.0
"""
from __future__ import annotations

import threading
from collections import deque
from statistics import mean
from typing import Deque, Iterable, List


class RingBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf: Deque[float] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # --------------------------------------------------------------------- #
    # mutating ops
    # --------------------------------------------------------------------- #
    def append(self, value: float) -> None:
        with self._lock:
            self._buf.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        with self._lock:
            self._buf.extend(map(float, values))

    # --------------------------------------------------------------------- #
    # stats / helpers
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:  # len(rb)
        return len(self._buf)

    def mean(self) -> float:
        with self._lock:
            return mean(self._buf) if self._buf else 0.0

    def to_list(self) -> List[float]:
        with self._lock:
            return list(self._buf)
