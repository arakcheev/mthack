"""Положение вагона по пройденному пути: точка на линии маршрута в сетке MGRS или путь от старта.

Есть выставка по GNSS: вагон едет по линии своего направления, положение — точка линии в месте
«место старта + путь» (с поправкой сбросов на остановках, domain/stops.py), в метрах сетки MGRS; курс — направление линии; высота — профиль линии.
Линия построена по антенне GNSS; base_link сдвинут от неё на AntennaOffset (вперёд по ходу и вниз).
Выставки нет: относительная одометрия — x = путь от старта вдоль пути, y = z = 0 (условие это допускает).

Неопределённость положения: вдоль пути σ растёт с путём от последнего точно известного места — старта или
сброса на остановке (дрейф оценки пути), поперёк —
постоянная (точность линии карты). Числа — в параметрах узла, их происхождение — docs/ACCURACY.md.
"""
import math
from dataclasses import dataclass
from typing import Protocol

from tram_backup_odometry.domain.geodesy import MgrsGrid
from tram_backup_odometry.domain.track import LinePosition, TrackLine


@dataclass(frozen=True)
class PoseUncertainty:
    along_at_start_m: float        # σ вдоль пути на старте (место выставки, антенна против base_link)
    along_per_metre: float         # рост σ вдоль пути на метр пути (дрейф оценки пути, доля)
    across_m: float                # σ поперёк пути (линия карты против настоящего пути)
    vertical_m: float              # σ высоты


@dataclass(frozen=True)
class AntennaOffset:
    """Где base_link относительно антенны GNSS, по которой построена линия маршрута."""
    ahead_m: float                 # base_link впереди антенны по ходу, м
    below_m: float                 # base_link ниже антенны, м


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    yaw: float                     # рад, от оси x против часовой
    covariance_xyz: tuple          # 3 × 3 построчно, м²


class PoseSource(Protocol):
    frame_id: str

    def pose(self, travelled_m: float) -> Pose: ...


def _covariance(yaw: float, sigma_along: float, sigma_across: float, sigma_vertical: float) -> tuple:
    """Ковариация x, y, z: эллипс «вдоль × поперёк пути», повёрнутый на курс."""
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    along, across = sigma_along ** 2, sigma_across ** 2
    xx = along * cos_yaw ** 2 + across * sin_yaw ** 2
    yy = along * sin_yaw ** 2 + across * cos_yaw ** 2
    xy = (along - across) * cos_yaw * sin_yaw
    return (xx, xy, 0.0,
            xy, yy, 0.0,
            0.0, 0.0, sigma_vertical ** 2)


class MapPose:
    """Положение на линии маршрута в сетке MGRS."""

    def __init__(self, line: TrackLine, position: LinePosition, grid: MgrsGrid, antenna: AntennaOffset,
                 uncertainty: PoseUncertainty, frame_id: str):
        self._line = line
        self._position = position
        self._grid = grid
        self._antenna = antenna
        self._uncertainty = uncertainty
        self.frame_id = frame_id

    def pose(self, travelled_m: float) -> Pose:
        base_link_s = self._position.at(travelled_m) + self._antenna.ahead_m
        easting, northing, altitude, heading = self._line.point(base_link_s)
        x, y = self._grid.to_grid(easting, northing)
        u = self._uncertainty
        sigma_along = u.along_at_start_m + u.along_per_metre * self._position.since_anchor_m(travelled_m)
        return Pose(x, y, altitude - self._antenna.below_m, heading,
                    _covariance(heading, sigma_along, u.across_m, u.vertical_m))


class RelativePose:
    """Путь от старта по оси x: места старта нет."""

    def __init__(self, uncertainty: PoseUncertainty, frame_id: str):
        self._uncertainty = uncertainty
        self.frame_id = frame_id

    def pose(self, travelled_m: float) -> Pose:
        u = self._uncertainty
        sigma_along = u.along_per_metre * abs(travelled_m)
        return Pose(travelled_m, 0.0, 0.0, 0.0, _covariance(0.0, sigma_along, u.across_m, u.vertical_m))
