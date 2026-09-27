"""Оценка скорости, ускорения и пути трамвая: уравнение движения + проверенные показания колёс.

Один такт (tick):
1. Каждое новое показание колёс по порядку прихода: уравнение движения двигает оценку до момента показания
   (малым шагом, с ручкой, действовавшей в каждый момент), проверка решает, правдоподобно ли показание,
   и если да — фильтр Калмана поправляет скорость и множители привода η±.
2. Если новых показаний нет — уравнение продолжает оценку до «сейчас».
3. Пока колёса чистые, якорь тени переезжает ближе (Shadow.refresh).

Когда все тележки отброшены, оценка откатывается к тени — к последней проверенной точке, продолженной одним
уравнением. Уже выданный выход не переписывается.
"""
import math
from collections import Counter
from dataclasses import dataclass

from tram_backup_odometry.domain.motion import MotionIntegrator, MotionState, TrackProfile
from tram_backup_odometry.domain.shadow import Shadow
from tram_backup_odometry.domain.speed_filter import SpeedFilter
from tram_backup_odometry.domain.timeline import HandleHistory, SensorClock
from tram_backup_odometry.domain.traction import TractionModel
from tram_backup_odometry.domain.units import NS_PER_S, seconds_to_ns
from tram_backup_odometry.domain.wheel_check.checker import WheelChecker
from tram_backup_odometry.domain.wheel_check.types import Prediction, WheelReading

MAX_WHEEL_KMH = 100.0                   # показание выше — не показание (конструкционная скорость 71-911ЕМ — 75 км/ч)
PAIR_WINDOW_NS = seconds_to_ns(0.05)    # показания тележек ближе — одновременные (для старта)
# Метка ручки не может убежать вперёд «сейчас» больше чем на столько. Иначе одна метка «из будущего»
# навсегда закрыла бы историю ручки: сообщения с меткой старше последней в неё не пишутся.
MAX_HANDLE_LEAD_NS = seconds_to_ns(1.0)


@dataclass(frozen=True)
class WheelScale:
    """Масштаб датчика колёс: сколько км/ч показывает тахометр на 1 м/с настоящей скорости (своё у вагона)."""
    kmh_per_mps: float

    def speed(self, kmh: float) -> float:
        return kmh / self.kmh_per_mps


@dataclass(frozen=True)
class Estimate:
    stamp_ns: int              # метка оценки, часы датчиков колёс
    speed: float               # продольная скорость, м/с
    distance: float            # путь от старта, м
    acceleration: float        # ускорение по уравнению движения, м/с²
    speed_variance: float      # дисперсия скорости фильтра, (м/с)²
    fresh: bool                # были новые показания колёс с прошлого такта


@dataclass(frozen=True)
class Status:
    """Состояние проверки и модели для диагностики."""
    bogie_faults: dict                   # {тележка: BogieFault | None}
    eta_traction: float
    eta_brake: float
    rollbacks: int
    events: dict                         # {причина: сколько раз}
    rejected_on_input: int               # сообщений отброшено сразу: скорость не число, < 0 или > MAX_WHEEL_KMH; ручка не число


@dataclass(frozen=True)
class _WheelMessage:
    bogie: str
    stamp_ns: int
    kmh: float
    arrival_ns: int


class OdometryEstimator:
    """Вход: on_handle, on_wheel (в порядке прихода). Выход: tick(сейчас) → Estimate или None до первого показания."""

    def __init__(self, traction: TractionModel, integrator: MotionIntegrator, handle: HandleHistory, shadow: Shadow,
                 speed_filter: SpeedFilter, checker: WheelChecker, scale: WheelScale, step_s: float):
        self._traction = traction
        self._integrator = integrator
        self._handle = handle
        self._shadow = shadow
        self._filter = speed_filter
        self._checker = checker
        self._scale = scale
        self._step_s = step_s
        self._clock = SensorClock()
        self._inbox: list[_WheelMessage] = []
        self._state: MotionState | None = None
        self._acceleration = 0.0             # ускорение оценки на последнем шаге уравнения
        self._events: Counter = Counter()
        self._rollbacks = 0
        self._rejected_on_input = 0

    # --- вход ---

    def on_handle(self, stamp_ns: int, position: int, arrival_ns: int) -> None:
        """Положение ручки. Не число — отбрасывается; метка не дальше MAX_HANDLE_LEAD_NS вперёд «сейчас»."""
        if not math.isfinite(position):
            self._rejected_on_input += 1
            return
        if self._clock.started:
            stamp_ns = min(stamp_ns, self._clock.sensor_time(arrival_ns) + MAX_HANDLE_LEAD_NS)
        self._handle.add(stamp_ns, position)

    def on_wheel(self, bogie: str, stamp_ns: int, kmh: float, arrival_ns: int) -> None:
        """Показание тележки, км/ч. Не число, отрицательное или невозможно большое — отбрасывается сразу."""
        if math.isfinite(kmh) and 0.0 <= kmh <= MAX_WHEEL_KMH:
            self._inbox.append(_WheelMessage(bogie, stamp_ns, kmh, arrival_ns))
        else:
            self._rejected_on_input += 1

    def place_on_track(self, track: TrackProfile) -> None:
        """Выставка: место на маршруте стало известно — дальше уравнение учитывает уклон и кривизну."""
        self._integrator.use_track(track)

    # --- выход ---

    def tick(self, now_ns: int) -> Estimate | None:
        """now_ns — «сейчас» по часам прихода сообщений (часы узла)."""
        if self._state is None:
            return self._start() if self._inbox else None
        fresh = bool(self._inbox)
        for message in self._inbox:
            state_ns = self._state.time_ns
            measurement_ns = max(state_ns, self._clock.measurement_time(state_ns, message.stamp_ns, message.arrival_ns))
            self._advance(measurement_ns)
            self._measure(message.bogie, measurement_ns, message.kmh)
        self._inbox.clear()
        if not fresh:
            self._advance(self._clock.sensor_time(now_ns))
        if self._shadow.refresh(self._state, self._checker.clean()):
            self._checker.reset_drift()
        self._shadow.remember(self._state)
        self._handle.forget_before(self._shadow.oldest_time_ns)
        return self._estimate(fresh)

    @property
    def distance(self) -> float:
        """Путь от старта по текущей оценке, м (0 до первого показания)."""
        return self._state.distance if self._state is not None else 0.0

    def status(self) -> Status:
        state = self._state
        return Status(
            bogie_faults=self._checker.faults(),
            eta_traction=state.eta_traction if state else 1.0,
            eta_brake=state.eta_brake if state else 1.0,
            rollbacks=self._rollbacks,
            events=dict(self._events),
            rejected_on_input=self._rejected_on_input,
        )

    # --- внутреннее ---

    def _estimate(self, fresh: bool) -> Estimate:
        s = self._state
        return Estimate(s.time_ns, s.speed, s.distance, self._acceleration, self._filter.speed_variance, fresh)

    def _start(self) -> Estimate:
        """Первое показание: стартовая скорость — меньшая из одновременных показаний тележек."""
        newest = max(self._inbox, key=lambda m: m.stamp_ns)
        kmh = min(m.kmh for m in self._inbox if newest.stamp_ns - m.stamp_ns <= PAIR_WINDOW_NS)
        time_ns = self._clock.start(newest.stamp_ns, newest.arrival_ns)
        self._handle.discard_after(time_ns + MAX_HANDLE_LEAD_NS)
        speed = self._scale.speed(kmh)
        self._state = MotionState(time_ns, speed, self._traction.steady_force(self._handle.latest, speed), 0.0)
        self._shadow.start(self._state)
        self._inbox.clear()
        return self._estimate(True)

    def _advance(self, until_ns: int) -> None:
        state = self._state
        if until_ns <= state.time_ns:
            return
        dt_s = (until_ns - state.time_ns) / NS_PER_S
        result = self._integrator.run(state, until_ns, self._step_s)
        if result.last_acceleration is not None:
            self._acceleration = result.last_acceleration
        self._filter.predict(state, result, dt_s)
        self._shadow.advance(until_ns)

    def _measure(self, bogie: str, time_ns: int, kmh: float) -> None:
        state = self._state
        speed = self._scale.speed(kmh)
        measurement_variance = self._filter.measurement_variance(self._checker.bogies_online(bogie, time_ns))
        handle = self._integrator.handle_at(time_ns)
        prediction = Prediction(
            speed=state.speed,
            variance=self._filter.speed_variance,
            measurement_variance=measurement_variance,
            shadow_speed=self._shadow.speed,
            shadow_age_s=self._shadow.age_s(time_ns),
            handle=handle,
            model_acceleration=self._shadow.model_acceleration,
        )
        verdict = self._checker.judge(WheelReading(bogie, time_ns, speed), prediction)
        if verdict.reason is not None:
            self._events[verdict.reason.value] += 1

        if verdict.rollback and self._checker.all_rejected():   # пока есть исправная тележка, опора — она
            self._filter.restart_speed(self._shadow.restore(state))
            self._rollbacks += 1
        if verdict.accept:
            if verdict.widen:
                self._filter.widen_speed_variance((speed - state.speed) ** 2)
            settled = not self._checker.all_rejected() and self._checker.clean()
            self._filter.correct(state, speed, measurement_variance, handle, settled)
        if verdict.reanchor:
            self._shadow.anchor_at(state, state.time_ns)
            self._checker.reset_drift()
