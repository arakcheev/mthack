"""Время оценки: часы датчиков колёс и история положения ручки.

Оценка живёт по часам датчиков (метки header.stamp сообщений колёс), а не по часам прихода сообщений:
по этим меткам выход сопоставляется с правильными значениями. Часы прихода нужны, чтобы продолжать
оценку, когда новых показаний нет, и чтобы не верить метке, убежавшей вперёд.
"""
import bisect
from dataclasses import dataclass, field

from tram_backup_odometry.domain.units import seconds_to_ns

# За один такт метка показания может убежать вперёд «сейчас по приходу» не больше чем на столько.
# Так пачка задержанных сообщений в начале записи догоняется плавно, а метка «+1 с» не прыгает.
MAX_STAMP_LEAD_NS = seconds_to_ns(0.1)


@dataclass
class SensorClock:
    """Перевод «сейчас по часам прихода» во время датчиков.

    arrival_minus_stamp_ns — сдвиг «приход − метка» последнего принятого показания.
    """
    arrival_minus_stamp_ns: float | None = None

    @property
    def started(self) -> bool:
        return self.arrival_minus_stamp_ns is not None

    def start(self, stamp_ns: int, arrival_ns: int) -> int:
        self.arrival_minus_stamp_ns = float(arrival_ns - stamp_ns)
        return stamp_ns

    def measurement_time(self, estimate_time_ns: int, stamp_ns: int, arrival_ns: int) -> int:
        """Время показания по часам оценки: его метка, но не дальше «сейчас по приходу» + MAX_STAMP_LEAD_NS."""
        latest_allowed = max(estimate_time_ns, arrival_ns - self.arrival_minus_stamp_ns) + MAX_STAMP_LEAD_NS
        measurement_ns = min(stamp_ns, int(latest_allowed))
        self.arrival_minus_stamp_ns = float(arrival_ns - measurement_ns)
        return measurement_ns

    def sensor_time(self, arrival_clock_ns: int) -> int:
        """«Сейчас» по часам датчиков, когда новых показаний нет."""
        return int(arrival_clock_ns - self.arrival_minus_stamp_ns)


@dataclass
class HandleHistory:
    """Положение ручки во времени по меткам сообщений ручки.

    latest — последнее полученное положение (в порядке прихода). История хранит только сообщения с неубывающей
    меткой: сообщение с меткой старше последней в историю не попадает (время датчика иногда идёт назад).
    """
    latest: float = 0.0
    _stamps_ns: list = field(default_factory=list)
    _positions: list = field(default_factory=list)

    def add(self, stamp_ns: int, position: float) -> None:
        self.latest = float(position)
        if not self._stamps_ns or stamp_ns >= self._stamps_ns[-1]:
            self._stamps_ns.append(stamp_ns)
            self._positions.append(float(position))

    def at(self, time_ns: int) -> float:
        """Положение ручки, действовавшее в момент time_ns; 0 (нейтраль), если сообщений до него не было."""
        index = bisect.bisect_right(self._stamps_ns, time_ns) - 1
        return self._positions[index] if index >= 0 else 0.0

    def discard_after(self, time_ns: int) -> None:
        """Выбросить положения с меткой позже time_ns — метки «из будущего», пришедшие до старта оценки
        (тогда их ещё не с чем было сравнить)."""
        keep = bisect.bisect_right(self._stamps_ns, time_ns)
        del self._stamps_ns[keep:]
        del self._positions[keep:]

    def forget_before(self, time_ns: int, min_batch: int = 256) -> None:
        """Забыть положения, которые уже не понадобятся: все до time_ns, кроме последнего действовавшего в time_ns.

        Без этого история растёт всё время работы узла. Удаляем пачками, чтобы не сдвигать список на каждом такте.
        """
        stale = bisect.bisect_right(self._stamps_ns, time_ns) - 1
        if stale >= min_batch:
            del self._stamps_ns[:stale]
            del self._positions[:stale]

    def __len__(self) -> int:
        return len(self._stamps_ns)
