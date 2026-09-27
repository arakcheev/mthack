"""Замер задержки «вход → публикация»: от прихода первого входного сообщения, ещё не попавшего в выход,
до публикации оценки, в которую оно вошло. Часы — монотонные часы компьютера."""
from dataclasses import dataclass

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class LatencyStats:
    count: int
    mean_ms: float
    max_ms: float


class _Accumulator:
    def __init__(self):
        self.count, self.total_ns, self.max_ns = 0, 0, 0

    def add(self, latency_ns: int) -> None:
        self.count += 1
        self.total_ns += latency_ns
        self.max_ns = max(self.max_ns, latency_ns)

    def stats(self) -> LatencyStats:
        mean = self.total_ns / self.count / NS_PER_MS if self.count else 0.0
        return LatencyStats(self.count, mean, self.max_ns / NS_PER_MS)


class LatencyMeter:
    def __init__(self):
        self._pending_since_ns: int | None = None
        self._since_start = _Accumulator()
        self._period = _Accumulator()

    def received(self, now_ns: int) -> None:
        if self._pending_since_ns is None:
            self._pending_since_ns = now_ns

    def published(self, now_ns: int) -> None:
        if self._pending_since_ns is None:
            return
        latency = now_ns - self._pending_since_ns
        self._pending_since_ns = None
        self._since_start.add(latency)
        self._period.add(latency)

    def since_start(self) -> LatencyStats:
        return self._since_start.stats()

    def take_period(self) -> LatencyStats:
        """Задержка за период с прошлого вызова; период начинается заново."""
        stats = self._period.stats()
        self._period = _Accumulator()
        return stats
