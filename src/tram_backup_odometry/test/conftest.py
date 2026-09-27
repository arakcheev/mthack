"""Общие заготовки тестов: модель из пакета и простая прямая линия маршрута."""
import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:      # тесты запускаются и без установки пакета (pytest из любой папки)
    sys.path.insert(0, str(PACKAGE_ROOT))

from tram_backup_odometry import model_file
from tram_backup_odometry.domain.track import TrackLine

MODEL_PATH = PACKAGE_ROOT / "config" / "model.json"
NS = 1_000_000_000


@pytest.fixture(scope="session")
def model():
    return model_file.load(MODEL_PATH)


def straight_line(direction: int, east0: float, north0: float, heading_deg: float, length_m: float) -> TrackLine:
    """Прямая линия через 2 м: без уклона и кривизны, высота растёт на 1 м на 100 м."""
    s = np.arange(0.0, length_m + 1e-9, 2.0)
    h = np.radians(heading_deg)
    return TrackLine(direction, s, np.full_like(s, 0.01), np.zeros_like(s),
                     s, s, east0 + s * np.cos(h), north0 + s * np.sin(h), 150.0 + 0.01 * s)
