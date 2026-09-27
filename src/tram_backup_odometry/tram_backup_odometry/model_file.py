"""Чтение файла модели config/model.json: числа вагонов, фильтра, проверки колёс и карта маршрута.

Файл пишется один раз офлайн (подбор по рейсам обучения) и едет внутри пакета; узел только читает.
Числа в JSON хранятся полностью (repr float), после чтения они те же до последнего знака.
Знание о формате файла живёт только здесь.
"""
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from tram_backup_odometry.domain.estimator import WheelScale
from tram_backup_odometry.domain.shadow import ShadowConfig
from tram_backup_odometry.domain.speed_filter import FilterConfig
from tram_backup_odometry.domain.stops import StopResetConfig
from tram_backup_odometry.domain.track import TrackLine
from tram_backup_odometry.domain.traction import TractionParams
from tram_backup_odometry.domain.wheel_check.types import WheelCheckConfig

FORMAT = "tram_backup_odometry/model/1"
LINE_ARRAYS = ("s", "grade", "curvature", "geometry_s", "geometry_profile_s", "easting", "northing", "altitude", "stop_s")


@dataclass(frozen=True)
class VehicleModel:
    traction: TractionParams
    wheel_scale: WheelScale


@dataclass(frozen=True)
class Model:
    vehicles: dict[str, VehicleModel]
    filter: FilterConfig
    wheel_check: WheelCheckConfig
    step_s: float             # шаг уравнения движения оценки, с
    shadow: ShadowConfig
    utm_zone: int
    max_start_offset_m: float
    lines: dict[int, TrackLine]     # {направление: линия}
    stop_reset: StopResetConfig


def _build(cls, values: dict):
    """Датакласс из словаря: лишний или недостающий ключ — ошибка чтения, а не тихое значение по умолчанию."""
    names = {f.name for f in fields(cls)}
    if set(values) != names:
        raise ValueError(f"{cls.__name__}: лишние ключи {sorted(set(values) - names)}, "
                         f"нет ключей {sorted(names - set(values))}")
    return cls(**values)


def _traction(values: dict) -> TractionParams:
    table = tuple(tuple(float(x) for x in row) for row in values["force_table"])
    nodes = tuple(float(x) for x in values["speed_nodes"])
    return _build(TractionParams, {**values, "force_table": table, "speed_nodes": nodes})


def _line(direction: int, values: dict) -> TrackLine:
    arrays = {name: np.asarray(values[name], dtype=float) for name in LINE_ARRAYS}
    return TrackLine(direction=direction, **arrays)


def dump(model: Model, path: Path, source: str) -> None:
    """Записать модель (делает офлайн-подбор). source — откуда числа, одной строкой."""
    def plain(obj) -> dict:
        return {f.name: getattr(obj, f.name) for f in fields(obj)}

    doc = {
        "format": FORMAT,
        "source": source,
        "vehicles": {number: {"wheel_scale_kmh_per_mps": v.wheel_scale.kmh_per_mps,
                              "traction": {**plain(v.traction), "force_table": [list(r) for r in v.traction.force_table],
                                           "speed_nodes": list(v.traction.speed_nodes)}}
                     for number, v in model.vehicles.items()},
        "filter": plain(model.filter),
        "wheel_check": {**plain(model.wheel_check), "rail_brake_notches": list(model.wheel_check.rail_brake_notches)},
        "estimator": {"step_s": model.step_s},
        "shadow": plain(model.shadow),
        "track": {"utm_zone": model.utm_zone, "max_start_offset_m": model.max_start_offset_m,
                  "lines": {str(d): {name: getattr(line, name).tolist() for name in LINE_ARRAYS}
                            for d, line in model.lines.items()}},
        "stop_reset": plain(model.stop_reset),
    }
    Path(path).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def load(path: Path) -> Model:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: формат {doc.get('format')!r}, ожидается {FORMAT!r}")
    check = dict(doc["wheel_check"])
    check["rail_brake_notches"] = tuple(check["rail_brake_notches"])
    return Model(
        vehicles={number: VehicleModel(_traction(v["traction"]), WheelScale(float(v["wheel_scale_kmh_per_mps"])))
                  for number, v in doc["vehicles"].items()},
        filter=_build(FilterConfig, doc["filter"]),
        wheel_check=_build(WheelCheckConfig, check),
        step_s=float(doc["estimator"]["step_s"]),
        shadow=_build(ShadowConfig, doc["shadow"]),
        utm_zone=int(doc["track"]["utm_zone"]),
        max_start_offset_m=float(doc["track"]["max_start_offset_m"]),
        lines={int(d): _line(int(d), line) for d, line in doc["track"]["lines"].items()},
        stop_reset=_build(StopResetConfig, doc["stop_reset"]),
    )
