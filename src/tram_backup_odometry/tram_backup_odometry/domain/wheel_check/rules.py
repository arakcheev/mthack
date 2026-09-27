"""Правила проверки показания колёс. Каждое правило — отдельный класс; проверка (WheelChecker) спрашивает
их по порядку, первое правило с ответом решает судьбу показания. Правило без ответа возвращает None.

Порядок важен и совпадает с порядком в WHEEL_RULES:
1. застыл датчик;  2. тележка отброшена — ждём её возврата;  3. одиночный выброс;
4. ускорение, которого у вагона не бывает;  5. буксование и юз;  6. медленное расхождение с уравнением;
7. расхождение тележек между собой.
"""
import math
from dataclasses import dataclass
from typing import Protocol

from tram_backup_odometry.domain.units import KMH_PER_MPS, NS_PER_S
from tram_backup_odometry.domain.wheel_check.bogie import Bogies, BogieTrack
from tram_backup_odometry.domain.wheel_check.types import (
    SKIP, BogieFault, Prediction, Recovery, Verdict, WheelCheckConfig, WheelReading,
)

MOVING_MPS = 0.3          # застывание ищем только на ходу: у стоящего вагона колесо и должно стоять


@dataclass(frozen=True)
class CheckContext:
    """Показание и всё, что о нём уже посчитано: общее для всех правил."""
    reading: WheelReading
    prediction: Prediction
    bogie: BogieTrack
    bogies: Bogies
    other_speed: float                   # скорость другой тележки в тот же момент; NaN — нет
    wheel_acceleration: float | None     # ускорение колеса за окно, м/с²; None — показаний мало
    spread: float                        # σ разности «показание − прогноз»: √(P + R)
    innovation: float                    # показание − прогноз фильтра, м/с
    shadow_departure: float              # показание − уравнение без колёс, м/с
    notch: int                           # ручка, целое положение


class WheelRule(Protocol):
    def judge(self, ctx: CheckContext) -> Verdict | None: ...


class FrozenSensorRule:
    """Значение не меняется дольше frozen_s на ходу, а уравнение или другая тележка изменились — датчик застыл."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        c, bogie, speed, time_ns = self._c, ctx.bogie, ctx.reading.speed, ctx.reading.time_ns
        if not (abs(speed - bogie.frozen_value) * KMH_PER_MPS < c.frozen_kmh):
            # значение сдвинулось: запоминаем новое и обстановку в этот момент
            bogie.frozen_value, bogie.frozen_since_ns = speed, time_ns
            bogie.frozen_model_speed, bogie.frozen_other_speed = ctx.prediction.speed, ctx.other_speed
        elif (time_ns - bogie.frozen_since_ns > c.frozen_s * NS_PER_S and bogie.fault is not BogieFault.FROZEN
              and speed > MOVING_MPS):
            model_moved = abs(ctx.prediction.speed - bogie.frozen_model_speed) > c.frozen_model_change_mps
            other_moved = abs(ctx.other_speed - bogie.frozen_other_speed) > c.frozen_model_change_mps
            if model_moved or other_moved:
                bogie.reject(time_ns, BogieFault.FROZEN, 0.0)
                return Verdict(False, rollback=True, reason=BogieFault.FROZEN)
        return None


class RejectedBogieRule:
    """Тележка отброшена: её показания не принимаются, пока она не вернётся. Возврат:
    - конец буксования или юза: колесо резко идёт обратно к прогнозу (защита отпустила колесо) и отклонение
      упало вдвое от наибольшего — такого скачка у самого вагона не бывает;
    - показания держатся у прогноза дольше return_hold_s;
    - страховка: все тележки отброшены давно, а между собой согласны — скорее неправо уравнение."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        c, bogie, time_ns = self._c, ctx.bogie, ctx.reading.time_ns
        if bogie.fault is None:
            return None
        if bogie.fault is BogieFault.FROZEN and abs(ctx.reading.speed - bogie.frozen_value) * KMH_PER_MPS < c.frozen_kmh:
            return SKIP

        bogie.max_departure = max(bogie.max_departure, abs(ctx.innovation))
        acceleration = ctx.wheel_acceleration
        if acceleration is not None and bogie.fault_direction and abs(ctx.innovation) < 0.5 * bogie.max_departure:
            excess = acceleration - bogie.mean_model_acceleration()
            slip_ends = bogie.fault_direction < 0 and excess > c.slip_excess_accel     # колесо раскручивается обратно
            slide_ends = bogie.fault_direction > 0 and excess < -c.slide_excess_decel  # колесо тормозится обратно
            if slip_ends or slide_ends:
                bogie.fault = None
                return Verdict(True, widen=True, reanchor=not ctx.bogies.all_rejected(), reason=Recovery.FAULT_ENDED)

        if abs(ctx.innovation) <= c.return_sigmas * ctx.spread:
            if bogie.returning_since_ns is None:
                bogie.returning_since_ns = time_ns
            if time_ns - bogie.returning_since_ns >= c.return_hold_s * NS_PER_S:
                bogie.fault = None
                return Verdict(True, reanchor=not ctx.bogies.all_rejected(), reason=Recovery.RETURNED)
        else:
            bogie.returning_since_ns = None

        all_drift = all(t.fault is not None and t.fault.is_slow_drift for t in ctx.bogies.all())
        wait_s = c.fallback_trust_drift_s if all_drift else c.fallback_trust_s
        rejected_for = time_ns - max(t.fault_time_ns for t in ctx.bogies.all())
        bogies_agree = math.isfinite(ctx.other_speed) and abs(ctx.reading.speed - ctx.other_speed) < c.bogie_disagreement_mps
        if ctx.bogies.all_rejected() and rejected_for >= wait_s * NS_PER_S and bogies_agree:
            for track in ctx.bogies.all():
                track.fault = None
            return Verdict(True, widen=True, reanchor=True, reason=Recovery.FALLBACK_TRUST)
        return SKIP


class OutlierRule:
    """Показание далеко от прогноза: одиночное пропустить; выбросы подряд дольше outlier_run_s — отбросить тележку."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        c, bogie, time_ns = self._c, ctx.bogie, ctx.reading.time_ns
        if abs(ctx.innovation) <= c.outlier_sigmas * ctx.spread + c.outlier_floor_mps:
            bogie.outlier_since_ns = None
            return None
        if bogie.outlier_since_ns is None:
            bogie.outlier_since_ns = time_ns
        if time_ns - bogie.outlier_since_ns > c.outlier_run_s * NS_PER_S:
            bogie.reject(time_ns, BogieFault.OUTLIER, ctx.innovation)
            return Verdict(False, rollback=True, reason=BogieFault.OUTLIER)
        return SKIP


class ImpossibleAccelerationRule:
    """Колесо разгоняется быстрее max_wheel_accel или тормозит сильнее min_wheel_accel и уже ушло от уравнения.
    При ручках rail_brake_notches тормозного момента на колёсах нет, и замедление любой силы — настоящее
    (рельсовый тормоз), поэтому там сильное замедление не отбрасывается."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        c, acceleration, departure = self._c, ctx.wheel_acceleration, ctx.shadow_departure
        if acceleration is None:
            return None
        too_fast = acceleration > c.max_wheel_accel and departure > c.min_departure_mps
        too_hard = (acceleration < c.min_wheel_accel and departure < -c.min_departure_mps
                    and ctx.notch not in c.rail_brake_notches)
        if too_fast or too_hard:
            ctx.bogie.reject(ctx.reading.time_ns, BogieFault.IMPOSSIBLE_ACCELERATION, departure)
            return Verdict(False, rollback=True, reason=BogieFault.IMPOSSIBLE_ACCELERATION)
        return None


class SlipSlideRule:
    """Колесо разгоняется быстрее уравнения больше чем на slip_excess_accel при ручке ≥ 0 и ушло вверх —
    буксование. Тормозит быстрее больше чем на slide_excess_decel (кроме ручек рельсового тормоза) и ушло
    вниз — юз при торможении, занижение при тяге."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        c, acceleration, departure = self._c, ctx.wheel_acceleration, ctx.shadow_departure
        if acceleration is None:
            return None
        excess = acceleration - ctx.bogie.mean_model_acceleration()
        if excess > c.slip_excess_accel and ctx.notch >= 0 and departure > c.min_departure_mps:
            ctx.bogie.reject(ctx.reading.time_ns, BogieFault.SLIP, 1)
            return Verdict(False, rollback=True, reason=BogieFault.SLIP)
        if excess < -c.slide_excess_decel and ctx.notch not in c.rail_brake_notches and departure < -c.min_departure_mps:
            fault = BogieFault.SLIDE if ctx.notch < 0 else BogieFault.UNDER_READING
            ctx.bogie.reject(ctx.reading.time_ns, fault, -1)
            return Verdict(False, rollback=True, reason=fault)
        return None


class SlowDriftRule:
    """Медленное расхождение колёс с уравнением без колёс — CUSUM в единицах σ расхождения (σ растёт с возрастом
    якоря). Считается только при ручке ≥ 0: при торможении уравнение ошибается сильнее, по уровню там не судим.
    Срабатывание не откатывает оценку: медленное расхождение — чаще ошибка уравнения, чем колёс.
    Занижение при ручках рельсового тормоза — настоящее торможение: показание принимается."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        c, bogie = self._c, ctx.bogie
        sigma = min(c.drift_sigma_max_mps,
                    math.sqrt(c.drift_sigma0_mps ** 2 + (c.drift_sigma_growth_mps_per_s * ctx.prediction.shadow_age_s) ** 2))
        normalized = (ctx.reading.speed - ctx.prediction.shadow_speed) / sigma
        pulling = ctx.notch >= 0
        bogie.cusum_up = max(0.0, bogie.cusum_up + normalized - c.cusum_allowance) if pulling else 0.0
        bogie.cusum_down = max(0.0, bogie.cusum_down - normalized - c.cusum_allowance) if pulling else 0.0
        if bogie.cusum_up > c.cusum_threshold:
            bogie.reject(ctx.reading.time_ns, BogieFault.DRIFT_UP, 1)
            return Verdict(False, reason=BogieFault.DRIFT_UP)
        if bogie.cusum_down > c.cusum_threshold:
            if ctx.notch in c.rail_brake_notches:
                bogie.cusum_down = 0.0
                return Verdict(True, reanchor=True, reason=Recovery.RAIL_BRAKE)
            bogie.reject(ctx.reading.time_ns, BogieFault.DRIFT_DOWN, -1)
            return Verdict(False, reason=BogieFault.DRIFT_DOWN)
        return None


class BogieDisagreementRule:
    """Тележки расходятся больше bogie_disagreement_mps — одна врёт. При тяге врёт бо́льшая (буксует),
    при торможении — меньшая (юз), на выбеге — та, что дальше от прогноза. Отбрасывается она."""

    def __init__(self, config: WheelCheckConfig):
        self._c = config

    def judge(self, ctx: CheckContext) -> Verdict | None:
        other = ctx.bogies.other(ctx.reading.bogie)
        speed, other_speed = ctx.reading.speed, ctx.other_speed
        if other is None or other.fault is not None or not math.isfinite(other_speed):
            return None
        if abs(speed - other_speed) <= self._c.bogie_disagreement_mps:
            return None
        if ctx.notch > 0:
            this_is_wrong = speed > other_speed
        elif ctx.notch < 0:
            this_is_wrong = speed < other_speed
        else:
            this_is_wrong = abs(ctx.innovation) > abs(other_speed - ctx.prediction.speed)
        if this_is_wrong:
            ctx.bogie.reject(ctx.reading.time_ns, BogieFault.DISAGREEMENT, speed - other_speed)
            return Verdict(False, reason=BogieFault.DISAGREEMENT)
        other.reject(ctx.reading.time_ns, BogieFault.DISAGREEMENT, other_speed - speed)
        return None


WHEEL_RULES = (FrozenSensorRule, RejectedBogieRule, OutlierRule, ImpossibleAccelerationRule,
               SlipSlideRule, SlowDriftRule, BogieDisagreementRule)
