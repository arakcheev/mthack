"""Фильтр Калмана по состоянию x = [v, η+, η−].

v — скорость вагона, η± — множители силы тяги и торможения. Прогноз делает уравнение движения
(MotionIntegrator); здесь — только ковариация P и поправка по показанию колёс.

η± между поправками тянутся к 1 (процесс Орнштейна — Уленбека с временем забывания tau_eta_s):
подстройка под износ и загрузку не накапливается навсегда. η подстраивается только в своём режиме
(тяга или торможение), и только пока колёса чистые.
"""
import math
from dataclasses import dataclass

import numpy as np

from tram_backup_odometry.domain.motion import MAX_SPEED_MPS, IntegrationResult, MotionState


@dataclass(frozen=True)
class FilterConfig:
    wheel_noise_mps: float     # σ_w — шум показания одной тележки, м/с
    speed_noise: float         # q_a — рост дисперсии скорости уравнения, м²/с³
    eta_spread: float          # σ_η — разброс η между рейсами
    eta_memory_s: float        # τ_η — за сколько секунд η забывает подстройку
    min_drive_force: float     # f̃, ниже которого η не подстраивается (привод почти выключен)
    eta_min: float
    eta_max: float


class SpeedFilter:
    def __init__(self, config: FilterConfig):
        self._config = config
        eta_variance = config.eta_spread ** 2
        self._covariance = np.diag([config.wheel_noise_mps ** 2, eta_variance, eta_variance])

    @property
    def speed_variance(self) -> float:
        return float(self._covariance[0, 0])

    def predict(self, state: MotionState, integration: IntegrationResult, dt_s: float) -> None:
        """Ковариация после шага уравнения длиной dt_s; η± тянутся к 1."""
        c = self._config
        eta_keep = math.exp(-dt_s / c.eta_memory_s)
        eta_noise = c.eta_spread ** 2 * (1 - eta_keep * eta_keep)
        transition = np.array([[1.0, integration.speed_per_eta_traction, integration.speed_per_eta_brake],
                               [0.0, eta_keep, 0.0],
                               [0.0, 0.0, eta_keep]])
        self._covariance = (transition @ self._covariance @ transition.T
                            + np.diag([c.speed_noise * dt_s, eta_noise, eta_noise]))
        state.eta_traction = 1.0 + eta_keep * (state.eta_traction - 1.0)
        state.eta_brake = 1.0 + eta_keep * (state.eta_brake - 1.0)

    def measurement_variance(self, bogies_online: int) -> float:
        """Дисперсия показания одной тележки. Две тележки видят одну и ту же скорость вагона, и две поправки
        подряд не должны удваивать вес колёс против уравнения — поэтому при двух тележках дисперсия вдвое больше."""
        return self._config.wheel_noise_mps ** 2 * (2.0 if bogies_online >= 2 else 1.0)

    def correct(self, state: MotionState, measured_speed: float, measurement_variance: float,
                handle: float, wheels_settled: bool) -> None:
        """Поправка по показанию колёс. η+ подстраивается только при тяге, η− — только при торможении,
        и только когда привод действительно тянет или тормозит, а колёса чистые (wheels_settled)."""
        c = self._config
        tune_traction = handle > 0 and state.force > c.min_drive_force and wheels_settled
        tune_brake = handle < 0 and state.force < -c.min_drive_force and wheels_settled
        covariance = self._covariance
        gain = covariance[:, 0] / (covariance[0, 0] + measurement_variance)
        gain[1] *= tune_traction
        gain[2] *= tune_brake
        innovation = measured_speed - state.speed
        state.speed = min(max(0.0, state.speed + gain[0] * innovation), MAX_SPEED_MPS)
        state.eta_traction = min(max(c.eta_min, state.eta_traction + gain[1] * innovation), c.eta_max)
        state.eta_brake = min(max(c.eta_min, state.eta_brake + gain[2] * innovation), c.eta_max)
        # форма Джозефа: P остаётся симметричной и положительной и при обнулённых компонентах усиления
        keep = np.eye(3)
        keep[:, 0] -= gain
        self._covariance = keep @ covariance @ keep.T + measurement_variance * np.outer(gain, gain)

    def widen_speed_variance(self, at_least: float) -> None:
        """Поверить показанию больше, чем уравнению: дисперсия скорости не меньше at_least."""
        self._covariance[0, 0] = max(self._covariance[0, 0], at_least)

    def restart_speed(self, model_age_s: float) -> None:
        """После отката к уравнению без колёс: дисперсия скорости — как у уравнения, идущего model_age_s секунд
        от проверенной точки; связь скорости с η± забыта."""
        c = self._config
        self._covariance[0, 0] = c.wheel_noise_mps ** 2 + c.speed_noise * model_age_s
        self._covariance[0, 1:] = self._covariance[1:, 0] = 0.0
