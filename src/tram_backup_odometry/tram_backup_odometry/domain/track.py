"""Карта линии маршрута: где вагон, какой там уклон и кривизна, куда смотрит путь.

Линия одного направления — точки через 2 м вдоль пути: s (м от начала линии), координаты UTM (восток,
север), высота, уклон, кривизна. Карта построена офлайн по GNSS рейсов обучения (docs/MODEL.md).

Место вагона на линии — s = start_s + пройденный путь. start_s находится один раз, при выставке по GNSS
в первые секунды; дальше GNSS не используется.
"""
import math
from dataclasses import dataclass, field

import numpy as np

MIN_TRACK_AHEAD_M = 300.0   # у линии впереди меньше пути — вагон на её конце, ехать по ней некуда


@dataclass(frozen=True)
class TrackLine:
    """Линия одного направления. Два набора узлов через 2 м:
    - профиль (s, grade, curvature) — для уравнения движения, по той части пути, где он подобран;
    - геометрия (geometry_s, easting, northing, altitude) — для положения; она длиннее профиля и заходит
      в разворотное кольцо за конечной, куда рейсы доезжают после конца профиля.
    Шкала геометрии — настоящая длина вдоль линии. Шкала профиля у концов линии от неё отличается на
    метры (сглаживание стянуло крайние точки); profile_s переводит одну в другую."""
    direction: int
    s: np.ndarray               # м, узлы профиля
    grade: np.ndarray           # доля: dh/ds, плюс — подъём по ходу
    curvature: np.ndarray       # 1/м
    geometry_s: np.ndarray      # м, узлы геометрии: длина вдоль линии
    geometry_profile_s: np.ndarray   # те же узлы в шкале профиля
    easting: np.ndarray         # UTM, м
    northing: np.ndarray        # UTM, м
    altitude: np.ndarray        # м, высота по GNSS (медиана рейсов)
    stop_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    # м, шкала длины: места, где рейсы обучения почти всегда встают в одной точке (сброс пути, domain/stops.py)

    def profile_s(self, geometry_s: float) -> float:
        """Место на линии в шкале профиля (для уклона и кривизны) по месту в шкале длины."""
        return float(np.interp(geometry_s, self.geometry_s, self.geometry_profile_s))

    def length_ahead(self, s: float) -> float:
        return float(self.geometry_s[-1] - s)

    def nearest(self, easting: float, northing: float) -> tuple[float, float]:
        """Ближайшая точка линии — проекция на ближайший отрезок: её место s (шкала длины) и расстояние до неё, м."""
        index = int(np.argmin(np.hypot(self.easting - easting, self.northing - northing)))
        best_s, best_distance = float(self.geometry_s[index]), math.inf
        for segment in range(max(index - 1, 0), min(index + 1, len(self.geometry_s) - 2) + 1):
            e0, n0 = self.easting[segment], self.northing[segment]
            de, dn = self.easting[segment + 1] - e0, self.northing[segment + 1] - n0
            fraction = min(max(((easting - e0) * de + (northing - n0) * dn) / (de * de + dn * dn), 0.0), 1.0)
            distance = math.hypot(easting - e0 - fraction * de, northing - n0 - fraction * dn)
            if distance < best_distance:
                s0, s1 = self.geometry_s[segment], self.geometry_s[segment + 1]
                best_s, best_distance = float(s0 + fraction * (s1 - s0)), distance
        return best_s, best_distance

    def point(self, s: float) -> tuple[float, float, float, float]:
        """Восток, север, высота и курс (рад, от оси «восток» против часовой) в месте s.
        За концами геометрии — её крайняя точка: куда вагон поехал дальше по путям конечной, неизвестно."""
        nodes = self.geometry_s
        s = min(max(s, float(nodes[0])), float(nodes[-1]))
        segment = min(max(int(np.searchsorted(nodes, s, side="right")) - 1, 0), len(nodes) - 2)
        s0, s1 = nodes[segment], nodes[segment + 1]
        fraction = (s - s0) / (s1 - s0)
        e0, e1 = self.easting[segment], self.easting[segment + 1]
        n0, n1 = self.northing[segment], self.northing[segment + 1]
        heading = math.atan2(n1 - n0, e1 - e0)
        altitude = float(np.interp(s, nodes, self.altitude))
        return float(e0 + (e1 - e0) * fraction), float(n0 + (n1 - n0) * fraction), altitude, heading


@dataclass(frozen=True)
class Placement:
    """Где вагон стоял на старте: направление, место на линии и как далеко от неё был фикс GNSS."""
    line: TrackLine
    start_s: float
    offset_m: float


class LinePosition:
    """Место антенны на линии (шкала длины): место старта по GNSS + пройденный путь + поправка от сбросов
    на остановках. Одно на оценку (уклон и кривизна под вагоном) и на положение для выхода."""

    def __init__(self, start_s: float):
        self.start_s = start_s
        self._correction_m = 0.0
        self._travelled_at_anchor_m = 0.0     # путь в момент, когда место было известно точно: старт или сброс

    @property
    def correction_m(self) -> float:
        return self._correction_m

    def at(self, travelled_m: float) -> float:
        return self.start_s + travelled_m + self._correction_m

    def since_anchor_m(self, travelled_m: float) -> float:
        """Путь от последнего точно известного места: от него копится ошибка пути."""
        return abs(travelled_m - self._travelled_at_anchor_m)

    def move_to(self, s: float, travelled_m: float) -> None:
        """Вагон сейчас (при пути travelled_m) стоит в месте s: дальнейшее место считается от него."""
        self._correction_m = s - self.start_s - travelled_m
        self._travelled_at_anchor_m = travelled_m


class PlacedTrack:
    """Уклон и кривизна в месте, куда вагон доехал от известного старта."""

    def __init__(self, line: TrackLine, start_s: float, position: LinePosition | None = None):
        """start_s — место старта в шкале профиля; position — поправки места от сбросов на остановках."""
        self._line = line
        self._start_s = start_s
        self._position = position

    def grade_and_curvature(self, travelled_m: float) -> tuple[float, float]:
        s = self._start_s + travelled_m + (self._position.correction_m if self._position is not None else 0.0)
        return float(np.interp(s, self._line.s, self._line.grade)), float(np.interp(s, self._line.s, self._line.curvature))


class FlatTrack:
    """Место неизвестно (нет GNSS на старте): путь считается ровным и прямым."""

    def grade_and_curvature(self, travelled_m: float) -> tuple[float, float]:
        return 0.0, 0.0


class TrackMap:
    """Линии обоих направлений маршрута."""

    def __init__(self, lines: dict[int, TrackLine], max_offset_m: float):
        self._lines = lines
        self._max_offset_m = max_offset_m

    def locate(self, easting: float, northing: float) -> Placement | None:
        """Место стоящего вагона по точке GNSS. Точка дальше max_offset_m от обеих линий — места нет.

        Направление по стоящему вагону не видно, а линии направлений идут рядом (соседние пути, 3–4 м).
        На конечной у линии «обратного» направления впереди почти нет пути — остаётся одно направление.
        Если впереди много пути по обеим линиям (старт посреди маршрута) — берём ближайшую линию.
        """
        candidates = []
        for line in self._lines.values():
            s, offset = line.nearest(easting, northing)
            if offset < self._max_offset_m:
                candidates.append(Placement(line, s, offset))
        if not candidates:
            return None
        with_track_ahead = [p for p in candidates if p.line.length_ahead(p.start_s) > MIN_TRACK_AHEAD_M]
        if len(with_track_ahead) > 1:
            return min(with_track_ahead, key=lambda p: p.offset_m)
        return max(candidates, key=lambda p: p.line.length_ahead(p.start_s))
