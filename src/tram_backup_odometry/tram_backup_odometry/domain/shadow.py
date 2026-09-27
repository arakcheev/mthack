"""Якорь и тень: опора, с которой проверка сравнивает колёса и к которой откатывается оценка.

Якорь — состояние оценки в прошлом, когда колёса были чистыми. Тень — то же уравнение движения без колёс,
продолженное от якоря до «сейчас». Пока колёса согласны с уравнением, якорь раз в refresh_s переезжает на
состояние lag_s назад: отставание нужно, чтобы начинающийся сбой колёс не успел попасть в якорь.
"""
from collections import deque
from dataclasses import dataclass

from tram_backup_odometry.domain.motion import MotionIntegrator, MotionState
from tram_backup_odometry.domain.units import NS_PER_S, seconds_to_ns


@dataclass(frozen=True)
class ShadowConfig:
    step_s: float              # шаг уравнения тени
    anchor_lag_s: float        # насколько якорь отстаёт от «сейчас»
    anchor_refresh_s: float    # как часто якорь переезжает


class Shadow:
    def __init__(self, integrator: MotionIntegrator, config: ShadowConfig):
        self._integrator = integrator
        self._config = config
        self._lag_ns = seconds_to_ns(config.anchor_lag_s)
        self._refresh_ns = seconds_to_ns(config.anchor_refresh_s)
        self._anchor: MotionState | None = None
        self._shadow: MotionState | None = None
        self._snapshots: deque = deque()     # состояния оценки на концах тактов — кандидаты в якорь
        self._next_refresh_ns = 0
        self.model_acceleration = 0.0       # ускорение тени на последнем шаге, м/с²

    def start(self, state: MotionState) -> None:
        self._anchor, self._shadow = state.copy(), state.copy()
        self._snapshots.append(state.copy())

    @property
    def speed(self) -> float:
        return self._shadow.speed

    def age_s(self, time_ns: int) -> float:
        """Сколько секунд тень идёт от якоря к моменту time_ns."""
        return (time_ns - self._anchor.time_ns) / NS_PER_S

    def advance(self, until_ns: int) -> None:
        result = self._integrator.run(self._shadow, until_ns, self._config.step_s)
        if result.last_acceleration is not None:
            self.model_acceleration = result.last_acceleration

    def anchor_at(self, snapshot: MotionState, now_ns: int) -> None:
        """Новый якорь; тень от него сразу догоняет «сейчас»."""
        self._anchor = snapshot.copy()
        self._shadow = snapshot.copy()
        self.advance(now_ns)

    def restore(self, state: MotionState) -> float:
        """Откат: оценка становится тенью (время не меняется). Возвращает возраст тени, с."""
        shadow = self._shadow
        state.speed, state.force, state.distance = shadow.speed, shadow.force, shadow.distance
        state.eta_traction, state.eta_brake = shadow.eta_traction, shadow.eta_brake
        return self.age_s(state.time_ns)

    def refresh(self, state: MotionState, wheels_clean: bool) -> bool:
        """Раз в anchor_refresh_s, если колёса чистые, — якорь на снимок lag_s назад. True — якорь переехал."""
        if state.time_ns < self._next_refresh_ns:
            return False
        self._next_refresh_ns = state.time_ns + self._refresh_ns
        if not wheels_clean:
            return False
        lagged_ns = state.time_ns - self._lag_ns
        snapshot = next((s for s in reversed(self._snapshots) if s.time_ns <= lagged_ns), None)
        if snapshot is None:
            return False
        self.anchor_at(snapshot, state.time_ns)
        return True

    def remember(self, state: MotionState) -> None:
        """Снимок оценки на конце такта. Хранятся только снимки, которые ещё могут стать якорем:
        из тех, что старше lag_s, нужен лишь самый новый (время оценки только растёт)."""
        self._snapshots.append(state.copy())
        lagged_ns = state.time_ns - self._lag_ns
        while len(self._snapshots) >= 2 and self._snapshots[1].time_ns <= lagged_ns:
            self._snapshots.popleft()

    @property
    def oldest_time_ns(self) -> int:
        """Самый ранний момент, от которого уравнение ещё может считаться заново (история ручки нужна с него)."""
        return min(self._anchor.time_ns, self._snapshots[0].time_ns)
