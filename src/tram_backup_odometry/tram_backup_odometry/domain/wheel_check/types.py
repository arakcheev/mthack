"""Что проверка колёс получает на вход, что отвечает и с какими настройками работает."""
import math
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class WheelCheckConfig:
    # одиночный выброс: |показание − прогноз| > outlier_sigmas·σ + outlier_floor_mps
    outlier_sigmas: float
    outlier_floor_mps: float
    outlier_run_s: float              # выбросы подряд дольше — тележка отброшена
    # медленное расхождение колёс с уравнением без колёс (CUSUM): σ расхождения растёт от якоря
    drift_sigma0_mps: float           # σ в момент якоря, м/с
    drift_sigma_growth_mps_per_s: float
    drift_sigma_max_mps: float
    cusum_allowance: float            # допуск на одно показание, в σ
    cusum_threshold: float            # порог срабатывания
    cusum_quiet_level: float          # ниже — колёса «чистые», якорь можно двигать
    # возврат отброшенной тележки
    return_sigmas: float              # |показание − прогноз| ≤ return_sigmas·σ …
    return_hold_s: float              # … дольше return_hold_s
    fallback_trust_s: float           # все тележки отброшены за скачок дольше — поверить согласным колёсам
    fallback_trust_drift_s: float     # … отброшены за медленное расхождение дольше
    # буксование, юз, невозможное ускорение, расхождение тележек
    bogie_disagreement_mps: float     # тележки расходятся больше — одна из них врёт
    slip_excess_accel: float          # колесо разгоняется быстрее уравнения больше чем на, м/с² — буксование
    slide_excess_decel: float         # колесо тормозит быстрее уравнения больше чем на, м/с² — юз
    min_departure_mps: float          # и колесо уже отошло от уравнения без колёс больше чем на, м/с
    max_wheel_accel: float            # ускорение колеса выше, м/с² — вагон так не разгоняется
    min_wheel_accel: float            # замедление колеса ниже, м/с² — вагон так не тормозит
    slope_window_s: float             # окно, по которому считается ускорение колеса
    # застывший датчик
    frozen_kmh: float                 # значение меняется меньше — «стоит»
    frozen_s: float                   # стоит дольше, а …
    frozen_model_change_mps: float    # … уравнение или другая тележка изменились больше — застыл
    rail_brake_notches: tuple         # ручки, при которых сильное замедление колёс — настоящее (рельсовый тормоз)


@dataclass(frozen=True)
class WheelReading:
    bogie: str            # "front" | "rear"
    time_ns: int          # по часам оценки
    speed: float          # м/с


@dataclass(frozen=True)
class Prediction:
    """Что ожидает модель в момент показания."""
    speed: float                   # прогноз фильтра до поправки, м/с
    variance: float                # его дисперсия, (м/с)²
    measurement_variance: float    # дисперсия показания, (м/с)²
    shadow_speed: float            # скорость по уравнению без колёс от якоря, м/с
    shadow_age_s: float            # сколько секунд уравнение без колёс идёт от якоря
    handle: float                  # ручка в момент показания
    model_acceleration: float      # ускорение уравнения без колёс в момент показания, м/с²


class BogieFault(Enum):
    """Почему показания тележки не принимаются."""
    FROZEN = "застыл датчик"
    OUTLIER = "выбросы подряд"
    IMPOSSIBLE_ACCELERATION = "ускорение, которого у вагона не бывает"
    SLIP = "буксование"
    SLIDE = "юз"
    UNDER_READING = "занижение без тормоза"
    DRIFT_UP = "медленно уходит вверх от уравнения"
    DRIFT_DOWN = "медленно уходит вниз от уравнения"
    DISAGREEMENT = "расходится с другой тележкой"

    @property
    def is_slow_drift(self) -> bool:
        return self in (BogieFault.DRIFT_UP, BogieFault.DRIFT_DOWN)


class Recovery(Enum):
    """Почему показание принято, хотя проверка его раньше не пускала или оно спорное."""
    FAULT_ENDED = "конец буксования или юза"
    RETURNED = "тележка вернулась к прогнозу"
    FALLBACK_TRUST = "все тележки отброшены давно, а между собой согласны — поверить колёсам"
    RAIL_BRAKE = "сильное торможение без тормозного момента — рельсовый тормоз"


@dataclass(frozen=True)
class Verdict:
    accept: bool                            # поправлять ли фильтр этим показанием
    rollback: bool = False                  # откатить оценку к уравнению без колёс
    reanchor: bool = False                  # после поправки поставить якорь на текущее состояние
    widen: bool = False                     # поверить показанию больше, чем уравнению
    reason: BogieFault | Recovery | None = None


ACCEPT = Verdict(True)
SKIP = Verdict(False)

NO_SPEED = math.nan   # скорость другой тележки неизвестна (нет свежего показания)
