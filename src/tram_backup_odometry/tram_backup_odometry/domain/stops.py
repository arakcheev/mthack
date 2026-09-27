"""Сброс ошибки пути на остановках.

Путь по колёсам копит ошибку: масштаб колёс меняется от даты к дате на доли процента, и к концу рейса место
на линии уходит на метры. Без GNSS его можно поправить только по тому, что известно о линии заранее.
Трамвай на части остановок встаёт почти в одной и той же точке: по GNSS рейсов обучения найдены места
линии, где стоят не меньше 60 % рейсов направления и 9 стоянок из 10 укладываются в 4 м (TrackLine.stop_s).

Вагон простоял confirm_s — это стоянка. Если рядом с его местом на линии есть такое место, вагон ставится
туда (LinePosition.move_to). «Рядом» — окно, которое растёт с путём от последнего точно известного места:
ошибка пути копится с путём. Одна стоянка — один сброс; стоянка на старте не сбрасывает (место там дал GNSS).
"""
from dataclasses import dataclass

import numpy as np

from tram_backup_odometry.domain.units import seconds_to_ns


@dataclass(frozen=True)
class StopResetConfig:
    still_speed_mps: float     # оценка скорости ниже — вагон стоит
    confirm_s: float           # столько простоять, чтобы считать это стоянкой, с
    window_m: float            # окно поиска места стоянки: постоянная часть, м
    window_per_metre: float    # рост окна на метр пути от последнего точно известного места (доля)


class StopReset:
    def __init__(self, stop_s: np.ndarray, config: StopResetConfig):
        self._stop_s = np.sort(np.asarray(stop_s, dtype=float))
        self._config = config
        self._confirm_ns = seconds_to_ns(config.confirm_s)
        self._still_since_ns: int | None = None
        self._this_stop_handled = True        # вагон стоит на старте: там место уже дал GNSS

    def observe(self, time_ns: int, speed_mps: float, line_s: float, since_anchor_m: float) -> float | None:
        """Такт оценки: время, скорость, место антенны на линии и путь от последнего точно известного места.
        Возвращает место стоянки, куда поставить вагон, или None."""
        config = self._config
        if speed_mps > config.still_speed_mps:
            self._still_since_ns = None
            self._this_stop_handled = False
            return None
        if self._still_since_ns is None:
            self._still_since_ns = time_ns
        if self._this_stop_handled or time_ns - self._still_since_ns < self._confirm_ns:
            return None
        self._this_stop_handled = True
        if not len(self._stop_s):
            return None
        nearest = float(self._stop_s[int(np.argmin(np.abs(self._stop_s - line_s)))])
        window = config.window_m + config.window_per_metre * since_anchor_m
        return nearest if abs(nearest - line_s) <= window else None
