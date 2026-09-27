"""Проверка показаний колёс перед тем, как пустить их в фильтр.

Колёса — измерение, не ответ: показание поправляет уравнение, только если правдоподобно. Проверка
сравнивает колесо с прогнозом фильтра и с «тенью» — уравнением без колёс от последней проверенной точки.
Так медленно нарастающая ошибка колёс не утаскивает за собой и опору сравнения.
"""
import math
from collections.abc import Iterable

from tram_backup_odometry.domain.units import seconds_to_ns
from tram_backup_odometry.domain.wheel_check.bogie import Bogies
from tram_backup_odometry.domain.wheel_check.rules import WHEEL_RULES, CheckContext, WheelRule
from tram_backup_odometry.domain.wheel_check.types import ACCEPT, Prediction, Verdict, WheelCheckConfig, WheelReading


class WheelChecker:
    def __init__(self, config: WheelCheckConfig, rules: Iterable[WheelRule] | None = None):
        self._config = config
        self._rules = tuple(rules) if rules is not None else tuple(rule(config) for rule in WHEEL_RULES)
        self._bogies = Bogies()
        self._window_ns = seconds_to_ns(config.slope_window_s)

    def judge(self, reading: WheelReading, prediction: Prediction) -> Verdict:
        bogie = self._bogies.track(reading.bogie)
        # скорость другой тележки — до того, как это показание запишется в историю
        other_speed = self._bogies.other_speed(reading.bogie, reading.time_ns)
        bogie.observe(reading.time_ns, reading.speed, prediction.model_acceleration, self._window_ns)
        ctx = CheckContext(
            reading=reading,
            prediction=prediction,
            bogie=bogie,
            bogies=self._bogies,
            other_speed=other_speed,
            wheel_acceleration=bogie.wheel_acceleration(self._config.slope_window_s),
            spread=math.sqrt(prediction.variance + prediction.measurement_variance),
            innovation=reading.speed - prediction.speed,
            shadow_departure=reading.speed - prediction.shadow_speed,
            notch=int(round(prediction.handle)),
        )
        for rule in self._rules:
            verdict = rule.judge(ctx)
            if verdict is not None:
                return verdict
        return ACCEPT

    def clean(self) -> bool:
        """Колёса чистые: ни одна тележка не отброшена и накопленное расхождение мало — якорь можно двигать."""
        quiet = self._config.cusum_quiet_level
        return all(t.fault is None and t.drift_quiet(quiet) for t in self._bogies.all())

    def all_rejected(self) -> bool:
        return self._bogies.all_rejected()

    def rejected_bogies(self) -> set[str]:
        return self._bogies.rejected_names()

    def bogies_online(self, bogie: str, time_ns: int) -> int:
        """Сколько исправных тележек на связи в момент показания тележки bogie (до его проверки)."""
        return self._bogies.online(bogie, time_ns)

    def reset_drift(self) -> None:
        """Новый якорь: расхождение с уравнением без колёс считается заново."""
        for track in self._bogies.all():
            track.forget_drift()

    def faults(self) -> dict:
        """Состояние тележек для диагностики: {тележка: причина отбрасывания или None}."""
        return {name: track.fault for name, track in self._bogies.items()}
