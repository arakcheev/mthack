"""Что проверка помнит об одной тележке и о паре тележек."""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tram_backup_odometry.domain.units import NS_PER_S, seconds_to_ns
from tram_backup_odometry.domain.wheel_check.types import NO_SPEED, BogieFault

SAMPLE_BUFFER = 64                    # показаний в окне ускорения — с запасом (тележка шлёт ~20 Гц)
FRESH_READING_NS = seconds_to_ns(0.2) # показание тележки «свежее» (одновременное с другим), если ближе


@dataclass
class BogieTrack:
    samples: deque = field(default_factory=lambda: deque(maxlen=SAMPLE_BUFFER))  # (время, скорость, ускорение уравнения)
    last_time_ns: int | None = None
    last_speed: float = NO_SPEED

    fault: BogieFault | None = None          # почему тележка отброшена; None — принимается
    fault_time_ns: int = 0
    fault_direction: int = 0                 # куда ушло колесо: +1 выше прогноза, −1 ниже, 0 — неизвестно
    max_departure: float = 0.0               # наибольшее |показание − прогноз|, пока тележка отброшена
    returning_since_ns: int | None = None    # с какого момента показания снова у прогноза
    outlier_since_ns: int | None = None      # с какого момента идут выбросы подряд
    cusum_up: float = 0.0                    # накопленное превышение над уравнением без колёс
    cusum_down: float = 0.0                  # накопленное занижение

    # застывание: значение, с какого момента оно стоит, прогноз и другая тележка в тот момент
    frozen_value: float = math.nan
    frozen_since_ns: int = 0
    frozen_model_speed: float = 0.0
    frozen_other_speed: float = math.nan

    def observe(self, time_ns: int, speed: float, model_acceleration: float, window_ns: int) -> None:
        self.samples.append((time_ns, speed, model_acceleration))
        while self.samples and self.samples[0][0] < time_ns - window_ns:
            self.samples.popleft()
        self.last_time_ns, self.last_speed = time_ns, speed

    def wheel_acceleration(self, window_s: float) -> float | None:
        """Ускорение колеса за окно — наклон прямой МНК по показаниям; None, если показаний мало."""
        points = list(self.samples)
        if len(points) < 4 or (points[-1][0] - points[0][0]) < 0.6 * window_s * NS_PER_S:
            return None
        times = np.array([p[0] for p in points], float) / NS_PER_S
        speeds = np.array([p[1] for p in points])
        times -= times.mean()
        spread = (times * times).sum()
        return float((times * (speeds - speeds.mean())).sum() / spread) if spread > 0 else None

    def mean_model_acceleration(self) -> float:
        return float(np.mean([p[2] for p in self.samples]))

    def drift_quiet(self, level: float) -> bool:
        """Накопленное расхождение с уравнением без колёс мало в обе стороны."""
        return self.cusum_up < level and self.cusum_down < level

    def forget_drift(self) -> None:
        self.cusum_up = self.cusum_down = 0.0

    def reject(self, time_ns: int, fault: BogieFault, departure: float) -> None:
        """Отбросить тележку. departure — куда ушло колесо от прогноза (знак), 0 — неизвестно."""
        self.fault, self.fault_time_ns = fault, time_ns
        self.returning_since_ns = self.outlier_since_ns = None
        self.forget_drift()
        self.fault_direction = int(math.copysign(1, departure)) if departure else 0
        self.max_departure = 0.0


class Bogies:
    """Обе тележки: кто из них сейчас отброшен."""

    def __init__(self):
        self._tracks: dict[str, BogieTrack] = {}

    def track(self, bogie: str) -> BogieTrack:
        return self._tracks.setdefault(bogie, BogieTrack())

    def other(self, bogie: str) -> BogieTrack | None:
        return next((t for name, t in self._tracks.items() if name != bogie), None)

    def other_speed(self, bogie: str, time_ns: int) -> float:
        """Скорость другой тележки, если её последнее показание почти одновременно с time_ns."""
        other = self.other(bogie)
        if other is None or other.last_time_ns is None or abs(other.last_time_ns - time_ns) >= FRESH_READING_NS:
            return NO_SPEED
        return other.last_speed

    def online(self, bogie: str, time_ns: int) -> int:
        """Сколько тележек сейчас на связи и не отброшено: эта (её показание пришло в time_ns) и те, чьё
        последнее показание свежее FRESH_READING_NS."""
        names = set(self._tracks) | {bogie}
        return sum(1 for name in names
                   if (name not in self._tracks or self._tracks[name].fault is None)
                   and (name == bogie or time_ns - self._tracks[name].last_time_ns < FRESH_READING_NS))

    def all(self):
        return self._tracks.values()

    def items(self):
        return self._tracks.items()

    def all_rejected(self) -> bool:
        return bool(self._tracks) and all(t.fault is not None for t in self._tracks.values())

    def rejected_names(self) -> set[str]:
        return {name for name, t in self._tracks.items() if t.fault is not None}
