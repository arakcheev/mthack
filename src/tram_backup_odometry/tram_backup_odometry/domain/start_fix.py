"""Выставка на старте: где стоит вагон, по GNSS первых секунд. После неё GNSS не используется.

Условие разрешает GNSS только для начальной выставки: в записи он либо есть первые секунды,
либо его нет совсем. Поэтому:
- первый годный фикс сразу даёт место (координаты конечны, до линии маршрута не дальше max_start_offset_m);
- следующие фиксы уточняют место медианой — только пока вагон стоит и не дольше collect_s от первого фикса;
- если вагон уже едет, место относится к пути, который оценка успела проехать к этому фиксу;
- если за wait_s от старта оценки годного фикса нет — выставки нет: дальше путь от старта без карты.
После этого решение не меняется, и узел отписывается от GNSS.
"""
import math
import statistics
from dataclasses import dataclass
from enum import Enum

from tram_backup_odometry.domain.geodesy import utm_forward
from tram_backup_odometry.domain.track import TrackLine, TrackMap
from tram_backup_odometry.domain.units import NS_PER_S

STANDING_M = 1.0      # вагон стоит, пока оценка проехала не больше этого от первого фикса


@dataclass(frozen=True)
class StartFixConfig:
    collect_s: float     # сколько секунд уточнять место от первого фикса
    wait_s: float        # сколько ждать первый годный фикс от старта оценки


@dataclass(frozen=True)
class GeoFix:
    latitude_deg: float
    longitude_deg: float


@dataclass(frozen=True)
class StartPlace:
    """Место старта: линия направления и s на ней, соответствующее пути оценки 0."""
    line: TrackLine
    route_s_at_zero: float
    offset_m: float           # расстояние фикса от линии, м


class AlignmentState(Enum):
    WAITING = "ждёт GNSS"
    REFINING = "место есть, уточняется"
    ALIGNED = "выставлена по GNSS"
    RELATIVE = "без выставки: путь от старта"


class StartAlignment:
    def __init__(self, track_map: TrackMap, utm_zone: int, config: StartFixConfig):
        self._track_map = track_map
        self._utm_zone = utm_zone
        self._config = config
        self.state = AlignmentState.WAITING
        self._fixes: list[GeoFix] = []
        self._first_fix_ns: int | None = None
        self._first_fix_travelled_m = 0.0
        self._started_ns: int | None = None

    @property
    def finished(self) -> bool:
        """Решение принято окончательно: GNSS больше не нужен."""
        return self.state in (AlignmentState.ALIGNED, AlignmentState.RELATIVE)

    def on_fix(self, fix: GeoFix, travelled_m: float, now_ns: int) -> StartPlace | None:
        """Фикс GNSS. Возвращает новое или уточнённое место старта; None — место не изменилось."""
        if self.finished or not (math.isfinite(fix.latitude_deg) and math.isfinite(fix.longitude_deg)):
            return None
        place = self._locate([*self._fixes, fix], travelled_m)
        if place is None:
            return None            # фикс далеко от маршрута — мусор
        if self._first_fix_ns is None:
            self._first_fix_ns, self._first_fix_travelled_m = now_ns, travelled_m
        self._fixes.append(fix)
        self.state = AlignmentState.REFINING
        return place

    def poll(self, travelled_m: float, now_ns: int) -> None:
        """Каждый такт после первого показания колёс: пора ли закончить выставку. Ожидание первого фикса
        отсчитывается от первого вызова — узел может быть запущен задолго до начала данных."""
        if self.finished:
            return
        if self._started_ns is None:
            self._started_ns = now_ns
        if self.state is AlignmentState.WAITING:
            if now_ns - self._started_ns >= self._config.wait_s * NS_PER_S:
                self.state = AlignmentState.RELATIVE
            return
        moved = travelled_m - self._first_fix_travelled_m > STANDING_M
        # уточнять не меньше collect_s и от первого фикса, и от первых данных колёс: фиксы могли прийти
        # задолго до колёс, и тогда один ранний фикс не должен сразу стать окончательным местом
        collecting_since = max(self._first_fix_ns, self._started_ns)
        if moved or now_ns - collecting_since >= self._config.collect_s * NS_PER_S:
            self.state = AlignmentState.ALIGNED

    def _locate(self, fixes: list[GeoFix], travelled_m: float) -> StartPlace | None:
        latitude = statistics.median(f.latitude_deg for f in fixes)
        longitude = statistics.median(f.longitude_deg for f in fixes)
        easting, northing = utm_forward(latitude, longitude, self._utm_zone)
        placement = self._track_map.locate(easting, northing)
        if placement is None:
            return None
        return StartPlace(placement.line, placement.start_s - travelled_m, placement.offset_m)
