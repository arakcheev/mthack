"""Одометрия трамвая целиком, без ROS: оценка движения + выставка по GNSS на старте + положение на карте
+ сброс ошибки пути на остановках.

Узел ROS только переводит сообщения в вызовы этого класса и обратно. Тот же класс гоняется офлайн по
записям — так проверяется ровно тот код, что работает в узле.
"""
from dataclasses import dataclass

from tram_backup_odometry.domain.estimator import Estimate, OdometryEstimator, Status
from tram_backup_odometry.domain.geodesy import MgrsGrid
from tram_backup_odometry.domain.pose import AntennaOffset, MapPose, Pose, PoseSource, PoseUncertainty, RelativePose
from tram_backup_odometry.domain.start_fix import AlignmentState, GeoFix, StartAlignment, StartPlace
from tram_backup_odometry.domain.stops import StopReset, StopResetConfig
from tram_backup_odometry.domain.track import LinePosition, PlacedTrack


@dataclass(frozen=True)
class PoseSettings:
    grid: MgrsGrid                 # квадрат MGRS, в котором выдаются x, y
    antenna: AntennaOffset         # base_link относительно антенны, по которой построена карта
    uncertainty: PoseUncertainty
    map_frame: str                 # frame_id положения на карте (MGRS)
    odom_frame: str                # frame_id относительной одометрии (без выставки)


@dataclass(frozen=True)
class Output:
    stamp_ns: int                  # метка выхода по часам эталона: метка колёс + задержка показания колёс
    estimate: Estimate
    pose: Pose | None              # None — место ещё не решено (ждём GNSS первые секунды)
    frame_id: str | None


@dataclass(frozen=True)
class AlignmentStatus:
    state: AlignmentState
    direction: int | None          # направление маршрута; None — места нет
    offset_m: float | None         # расстояние фикса GNSS от линии, м


class TramOdometry:
    def __init__(self, estimator: OdometryEstimator, alignment: StartAlignment, pose_settings: PoseSettings,
                 stop_reset: StopResetConfig, wheel_reading_delay_ns: int = 0):
        """wheel_reading_delay_ns — насколько показание колёс относится к моменту позже своей метки (тахометр ставит
        метку в начале окна счёта импульсов). Оценка живёт по меткам колёс, выход — по настоящему времени."""
        self._estimator = estimator
        self._wheel_reading_delay_ns = wheel_reading_delay_ns
        self._alignment = alignment
        self._settings = pose_settings
        self._stop_reset_config = stop_reset
        self._pose_source: PoseSource | None = None
        self._place: StartPlace | None = None
        self._line_position: LinePosition | None = None     # есть только при выставке на линию
        self._stop_reset: StopReset | None = None

    # --- вход ---

    def on_handle(self, stamp_ns: int, position: int, arrival_ns: int) -> None:
        self._estimator.on_handle(stamp_ns, position, arrival_ns)

    def on_wheel(self, bogie: str, stamp_ns: int, kmh: float, arrival_ns: int) -> None:
        self._estimator.on_wheel(bogie, stamp_ns, kmh, arrival_ns)

    def on_fix(self, fix: GeoFix, now_ns: int) -> None:
        """Фикс GNSS: используется только для выставки на старте, потом не принимается."""
        place = self._alignment.on_fix(fix, self._estimator.distance, now_ns)
        if place is None:
            return
        self._place = place
        position = LinePosition(place.route_s_at_zero)
        self._line_position = position
        self._stop_reset = StopReset(place.line.stop_s, self._stop_reset_config)
        self._estimator.place_on_track(PlacedTrack(place.line, place.line.profile_s(place.route_s_at_zero), position))
        s = self._settings
        self._pose_source = MapPose(place.line, position, s.grid, s.antenna, s.uncertainty, s.map_frame)

    @property
    def wants_gnss(self) -> bool:
        """Нужен ли ещё GNSS. False — выставка закончена, от GNSS можно отписаться."""
        return not self._alignment.finished

    # --- выход ---

    def tick(self, now_ns: int) -> Output | None:
        estimate = self._estimator.tick(now_ns)
        if estimate is None:
            return None            # данных колёс ещё нет: и ожидание GNSS ещё не началось
        self._alignment.poll(self._estimator.distance, now_ns)
        if self._alignment.state is AlignmentState.RELATIVE and self._pose_source is None:
            self._pose_source = RelativePose(self._settings.uncertainty, self._settings.odom_frame)
        self._reset_at_stop(estimate)
        stamp_ns = estimate.stamp_ns + self._wheel_reading_delay_ns
        if self._pose_source is None:
            return Output(stamp_ns, estimate, None, None)
        return Output(stamp_ns, estimate, self._pose_source.pose(estimate.distance), self._pose_source.frame_id)

    def _reset_at_stop(self, estimate: Estimate) -> None:
        position, reset = self._line_position, self._stop_reset
        if position is None or reset is None:
            return
        stop_s = reset.observe(estimate.stamp_ns, estimate.speed, position.at(estimate.distance),
                               position.since_anchor_m(estimate.distance))
        if stop_s is not None:
            position.move_to(stop_s, estimate.distance)

    def status(self) -> Status:
        return self._estimator.status()

    def alignment_status(self) -> AlignmentStatus:
        place = self._place
        return AlignmentStatus(self._alignment.state, place.line.direction if place else None,
                               place.offset_m if place else None)
