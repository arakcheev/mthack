"""Интегрирование уравнения движения малым шагом.

Состояние вагона двигается от своего времени до заданного момента шагами dt. На каждом шаге берутся
ручка, действовавшая в начале шага (по метке времени), уклон и кривизна в текущем месте пути, и
TractionModel.step даёт силу и скорость в конце шага. Путь — метод трапеций по скорости.

Множители η+ (тяга) и η− (торможение) поправляют силу привода: износ, загрузка вагона, состояние мотора.
При η = 1 шаг идёт ровно по уравнению. Интегратор возвращает, насколько скорость в конце отрезка зависит
от η± — это строка матрицы перехода фильтра (SpeedFilter.predict).
"""
from dataclasses import dataclass
from typing import Protocol

from tram_backup_odometry.domain.timeline import HandleHistory
from tram_backup_odometry.domain.traction import ROTATING_MASS_FACTOR, TractionModel
from tram_backup_odometry.domain.units import NS_PER_S, V_REF, seconds_to_ns

MAX_SPEED_MPS = 22.0   # уравнение не разгоняет вагон выше 79 км/ч (конструкционная скорость 71-911ЕМ — 75 км/ч)


@dataclass
class MotionState:
    """Состояние вагона по уравнению. Время — по часам датчиков."""
    time_ns: int
    speed: float                   # м/с
    force: float                   # f̃ — сила привода в долях веса
    distance: float                # путь от старта, м
    eta_traction: float = 1.0      # η+ — множитель силы тяги
    eta_brake: float = 1.0         # η− — множитель тормозной силы

    def copy(self) -> "MotionState":
        return MotionState(self.time_ns, self.speed, self.force, self.distance, self.eta_traction, self.eta_brake)


class TrackProfile(Protocol):
    """Уклон и кривизна пути в месте, куда вагон доехал от старта."""

    def grade_and_curvature(self, travelled_m: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class IntegrationResult:
    """Чувствительность скорости в конце отрезка к η+ и η− (м/с на единицу η) и ускорение на последнем шаге."""
    speed_per_eta_traction: float
    speed_per_eta_brake: float
    last_acceleration: float | None   # м/с²; None — шагов не было


class MotionIntegrator:
    """Двигает MotionState по уравнению движения до заданного момента."""

    def __init__(self, traction: TractionModel, track: TrackProfile, handle: HandleHistory):
        self._traction = traction
        self._track = track
        self._handle = handle

    def use_track(self, track: TrackProfile) -> None:
        """Место на маршруте стало известно (выставка по GNSS): дальше — с уклоном и кривизной."""
        self._track = track

    def handle_at(self, time_ns: int) -> float:
        return self._handle.at(time_ns)

    def run(self, state: MotionState, until_ns: int, step_s: float) -> IntegrationResult:
        gain_traction = gain_brake = 0.0
        last_acceleration = None
        step_ns = seconds_to_ns(step_s)
        while state.time_ns < until_ns:
            dt_ns = min(step_ns, until_ns - state.time_ns)
            dt_s = dt_ns / NS_PER_S
            handle = self._handle.at(state.time_ns)
            grade, curvature = self._track.grade_and_curvature(state.distance)
            step = self._traction.step(state.force, state.speed, handle, dt_s, grade, curvature)

            # скорость, которую добавляет или отнимает сила привода за шаг; η± масштабирует именно её
            drive_speed = step.force_impulse * V_REF / ROTATING_MASS_FACTOR
            traction = step.force_impulse > 0
            eta = state.eta_traction if traction else state.eta_brake
            speed = step.speed + (eta - 1.0) * drive_speed
            force = step.force
            if state.speed == 0.0 and handle <= 0:
                # стоящий вагон без тяги стоит: сопротивление не толкает его назад
                speed, force = 0.0, 0.0
            elif 0.0 < speed < MAX_SPEED_MPS:
                # на упоре (0 или MAX_SPEED_MPS) скорость от η не зависит
                if traction:
                    gain_traction += drive_speed
                else:
                    gain_brake += drive_speed
            speed = min(max(0.0, speed), MAX_SPEED_MPS)

            last_acceleration = (speed - state.speed) / dt_s
            state.distance += (state.speed + speed) / 2 * dt_s
            state.speed, state.force = speed, force
            state.time_ns += dt_ns
        return IntegrationResult(gain_traction, gain_brake, last_acceleration)
