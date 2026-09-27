"""Широта/долгота WGS84 → UTM и квадрат сетки MGRS.

Выход — декартовы координаты в сетке MGRS, как в Autoware: x, y — метры от юго-западного угла
квадрата 100 × 100 км. Вся карта держится в одном квадрате, поэтому x может выйти за 0…100 000
(маршрут №10 пересекает границу квадратов по восточной координате 400 000).

UTM — поперечная проекция Меркатора, ряды Крюгера до n⁴ (Karney, 2011, «Transverse Mercator with an
accuracy of a few nanometers»): ошибка меньше миллиметра в пределах зоны.
"""
import math
from dataclasses import dataclass

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
UTM_SCALE = 0.9996
UTM_FALSE_EASTING = 500_000.0

_N = WGS84_F / (2 - WGS84_F)
_RECTIFYING_RADIUS = WGS84_A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_ALPHA = (
    _N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16 + 41 * _N ** 4 / 180,
    13 * _N ** 2 / 48 - 3 * _N ** 3 / 5 + 557 * _N ** 4 / 1440,
    61 * _N ** 3 / 240 - 103 * _N ** 4 / 140,
    49561 * _N ** 4 / 161280,
)
_E_FACTOR = 2 * math.sqrt(_N) / (1 + _N)   # эксцентриситет в форме для конформной широты

_MGRS_COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
_MGRS_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"        # полосы широт по 8°, от 80° ю. ш.


def utm_forward(latitude_deg: float, longitude_deg: float, zone: int) -> tuple[float, float]:
    """Восток и север (м) точки в зоне UTM zone северного полушария."""
    phi = math.radians(latitude_deg)
    lam = math.radians(longitude_deg - (zone * 6 - 183))
    sin_phi = math.sin(phi)
    t = math.sinh(math.atanh(sin_phi) - _E_FACTOR * math.atanh(_E_FACTOR * sin_phi))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    easting, northing = eta, xi
    for j, alpha in enumerate(_ALPHA, start=1):
        easting += alpha * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
        northing += alpha * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
    return UTM_FALSE_EASTING + UTM_SCALE * _RECTIFYING_RADIUS * easting, UTM_SCALE * _RECTIFYING_RADIUS * northing


@dataclass(frozen=True)
class MgrsGrid:
    """Квадрат 100 × 100 км сетки MGRS: зона UTM и юго-западный угол квадрата в координатах UTM."""
    code: str
    zone: int
    origin_easting: float
    origin_northing: float

    @classmethod
    def parse(cls, code: str) -> "MgrsGrid":
        """Код квадрата: номер зоны, буква полосы, буква столбца, буква строки — например «37UDB».
        Только северное полушарие."""
        code = code.strip().upper()
        zone, band, column, row = int(code[:-3]), code[-3], code[-2], code[-1]
        if not 1 <= zone <= 60 or band not in _MGRS_BANDS or band < "N":
            raise ValueError(f"квадрат MGRS {code!r}: нужна зона 1–60 и полоса северного полушария N…X")
        columns = _MGRS_COLUMN_SETS[(zone - 1) % 3]
        if column not in columns or row not in _MGRS_ROW_LETTERS:
            raise ValueError(f"квадрат MGRS {code!r}: буквы столбца или строки не из сетки зоны {zone}")
        origin_easting = (columns.index(column) + 1) * 100_000.0
        # буквы строк повторяются каждые 2000 км; в чётных зонах ряд сдвинут на 5 букв
        row_northing = ((_MGRS_ROW_LETTERS.index(row) - (5 if zone % 2 == 0 else 0)) % 20) * 100_000.0
        band_south_lat = -80 + 8 * _MGRS_BANDS.index(band)
        _, band_south_northing = utm_forward(band_south_lat, zone * 6 - 183, zone)
        cycles = math.floor((band_south_northing - 100_000.0 - row_northing) / 2_000_000.0) + 1
        return cls(code, zone, origin_easting, row_northing + max(cycles, 0) * 2_000_000.0)

    def to_grid(self, easting: float, northing: float) -> tuple[float, float]:
        return easting - self.origin_easting, northing - self.origin_northing
