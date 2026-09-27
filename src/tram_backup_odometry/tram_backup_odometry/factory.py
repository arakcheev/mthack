"""Сборка оценки для одного вагона из файла модели — единственное место, где выбираются конкретные классы."""
from tram_backup_odometry.app import PoseSettings, TramOdometry
from tram_backup_odometry.domain.estimator import OdometryEstimator
from tram_backup_odometry.domain.motion import MotionIntegrator, TrackProfile
from tram_backup_odometry.domain.shadow import Shadow
from tram_backup_odometry.domain.speed_filter import SpeedFilter
from tram_backup_odometry.domain.start_fix import StartAlignment, StartFixConfig
from tram_backup_odometry.domain.timeline import HandleHistory
from tram_backup_odometry.domain.track import FlatTrack, TrackMap
from tram_backup_odometry.domain.traction import TractionModel
from tram_backup_odometry.domain.units import seconds_to_ns
from tram_backup_odometry.domain.wheel_check.checker import WheelChecker
from tram_backup_odometry.model_file import Model


class UnknownVehicle(ValueError):
    pass


def build_estimator(model: Model, vehicle: str, track: TrackProfile | None = None) -> OdometryEstimator:
    """Оценка для вагона vehicle. track — место на маршруте, если известно заранее; иначе путь ровный,
    пока выставка по GNSS не даст место (OdometryEstimator.place_on_track)."""
    if vehicle not in model.vehicles:
        raise UnknownVehicle(f"вагон {vehicle!r} не описан в модели; есть: {', '.join(sorted(model.vehicles))}")
    params = model.vehicles[vehicle]
    traction = TractionModel(params.traction)
    handle = HandleHistory()
    integrator = MotionIntegrator(traction, track if track is not None else FlatTrack(), handle)
    return OdometryEstimator(traction, integrator, handle, Shadow(integrator, model.shadow),
                             SpeedFilter(model.filter), WheelChecker(model.wheel_check), params.wheel_scale,
                             model.step_s)


def build_odometry(model: Model, vehicle: str, pose_settings: PoseSettings, start_fix: StartFixConfig,
                   wheel_reading_delay_s: float = 0.0) -> TramOdometry:
    """Одометрия целиком: оценка вагона + выставка по GNSS на старте + положение на карте."""
    alignment = StartAlignment(TrackMap(model.lines, model.max_start_offset_m), model.utm_zone, start_fix)
    return TramOdometry(build_estimator(model, vehicle), alignment, pose_settings, model.stop_reset,
                        seconds_to_ns(wheel_reading_delay_s))
