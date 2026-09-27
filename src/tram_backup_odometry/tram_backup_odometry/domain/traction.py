"""Тяговый привод и продольная динамика одного вагона — уравнение движения в безразмерных величинах.

    (1 + γ)·dṽ/dt̃ = f̃ − w̃₀(ṽ) − c_i·i(s) − c̃_к·κ̃(s)
    df̃/dt̃        = (F̃(u, ṽ) − f̃) / τ̃            (привод набирает силу с запаздыванием τ̃)
    w̃₀(ṽ)         = a + b·ṽ + c·ṽ²                (основное сопротивление движению)

ṽ = v / V_REF, t̃ = t / T_REF, κ̃ = L_REF / R. Силы — в долях веса вагона: f̃ = f / (m·g).
F̃(u, ṽ) — установившаяся сила привода при положении ручки u, таблица по опорным скоростям с линейной
интерполяцией (u = 0 — привод выключен). i — уклон (доля, плюс — подъём по ходу), κ = 1/R — кривизна пути.
Подробности и происхождение чисел — docs/MODEL.md.
"""
import bisect
import math
from dataclasses import dataclass

from tram_backup_odometry.domain.units import L_REF, MAX_HANDLE_NOTCH, T_REF, V_REF

ROTATING_MASS_FACTOR = 1.06   # 1 + γ: доля инерции вращающихся частей (колёса, редукторы, якоря моторов)


@dataclass(frozen=True)
class TractionParams:
    """Числа одного вагона, подобранные офлайн по рейсам обучения."""
    force_table: tuple        # F̃ по строкам ручки −15…−1, +1…+15 и столбцам speed_nodes
    speed_nodes: tuple        # опорные скорости ṽ таблицы, по возрастанию
    drive_lag: float          # τ̃ — безразмерное запаздывание силы привода; 0 — сила без запаздывания
    resistance_a: float       # a, b, c — основное сопротивление w̃₀ = a + b·ṽ + c·ṽ²
    resistance_b: float
    resistance_c: float
    grade_coef: float         # c_i — множитель уклона
    curve_coef: float         # c̃_к — множитель безразмерной кривизны


@dataclass(frozen=True)
class DriveStep:
    """Итог одного шага уравнения."""
    force: float              # f̃ в конце шага
    force_impulse: float      # ∫ f̃ dt̃ за шаг — по нему η± меняет скорость
    speed: float              # скорость в конце шага, м/с, при η± = 1


def _table_row(notch: int) -> int:
    """Номер строки таблицы для ручки: строки идут −15…−1, затем +1…+15 (нейтрали в таблице нет)."""
    return notch + MAX_HANDLE_NOTCH if notch < 0 else notch + MAX_HANDLE_NOTCH - 1


class TractionModel:
    """Сила привода по ручке и скорости и шаг уравнения движения."""

    def __init__(self, params: TractionParams):
        self._params = params

    def steady_force(self, handle: float, speed_mps: float) -> float:
        """F̃ при ручке handle и скорости speed_mps (м/с)."""
        return self._steady_force(handle, speed_mps / V_REF)

    def _steady_force(self, handle: float, speed_nd: float) -> float:
        notch = int(round(handle))
        if notch == 0:
            return 0.0
        notch = max(-MAX_HANDLE_NOTCH, min(MAX_HANDLE_NOTCH, notch))
        row = self._params.force_table[_table_row(notch)]
        nodes = self._params.speed_nodes
        speed = min(max(speed_nd, nodes[0]), nodes[-1])
        left = min(bisect.bisect_right(nodes, speed) - 1, len(nodes) - 2)
        weight = (speed - nodes[left]) / (nodes[left + 1] - nodes[left])
        return float(row[left] * (1 - weight) + row[left + 1] * weight)

    def step(self, force: float, speed_mps: float, handle: float, dt_s: float,
             grade: float, curvature_per_m: float) -> DriveStep:
        """Один шаг уравнения: сила привода force (f̃) и скорость speed_mps → их значения через dt_s секунд."""
        p = self._params
        speed_nd, dt_nd, curvature_nd = speed_mps / V_REF, dt_s / T_REF, curvature_per_m * L_REF
        target = self._steady_force(handle, speed_nd)
        if p.drive_lag > 0:
            # точное решение df̃/dt̃ = (F̃ − f̃)/τ̃ при постоянной F̃ на шаге
            decay = math.exp(-dt_nd / p.drive_lag)
            impulse = target * dt_nd + (force - target) * p.drive_lag * (1 - decay)
            force_end = target + (force - target) * decay
        else:
            force_end = target
            impulse = target * dt_nd
        basic = p.resistance_a + p.resistance_b * speed_nd + p.resistance_c * speed_nd * speed_nd
        resistance = basic + p.grade_coef * grade + p.curve_coef * curvature_nd
        speed_end = (speed_nd + (impulse - resistance * dt_nd) / ROTATING_MASS_FACTOR) * V_REF
        return DriveStep(force_end, impulse, speed_end)
