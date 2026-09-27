"""Оценка точности по записи: выход решения против GNSS того же рейса. Только для проверки — в узел GNSS не идёт.

    ros2 run tram_backup_odometry evaluate --input <bag рейса> --result <bag с /result/*>

Метрики как в условии (критерии 1 и 2):
- скорость: RMSE, MAE и среднее смещение (bias) /result/velocity против горизонтальной скорости GNSS master
  (вертикальная составляющая GNSS шумит), отдельно смещение при разгоне, торможении, выбеге и на стоянке
  (по ручке контроллера);
- путь: эталон — интеграл скорости GNSS, оценка — интеграл /result/velocity; ошибка вдоль пути
  MEAN / MAX / RMSE и дрейф в конце в % от пути;
- положение: расстояние x, y (и z) /result/position до фикса GNSS master, переведённого в тот же квадрат MGRS
  и перенесённого из антенны в base_link (вперёд по курсу на --antenna-behind, вниз на --antenna-above);
- частота выхода и задержка «вход → публикация» из /diagnostics, если он записан.
Пары «оценка — эталон» — по ближайшей метке header.stamp в пределах 0,05 с.
"""
import argparse
import math

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from tram_backup_odometry.domain.geodesy import MgrsGrid, utm_forward
from tram_backup_odometry.domain.units import KMH_PER_MPS, NS_PER_S
from tram_backup_odometry.ros.defaults import node_defaults

MATCH_TOLERANCE_S = 0.05
STOP_MPS = 0.3


def read_bag(path: str, topics: set) -> dict:
    """{топик: [(время записи нс, сообщение), ...]} для нужных топиков."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types() if t.name in topics}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    out = {name: [] for name in types}
    while reader.has_next():
        name, data, t_ns = reader.read_next()
        out[name].append((t_ns, deserialize_message(data, types[name])))
    return out


def stamp_s(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec / NS_PER_S


def nearest(ref_t: np.ndarray, est_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой точки эталона — индекс ближайшей оценки; маска пар в пределах допуска."""
    order = np.argsort(est_t)
    est_sorted = est_t[order]
    right = np.clip(np.searchsorted(est_sorted, ref_t), 1, len(est_sorted) - 1)
    left = right - 1
    pick = np.where(np.abs(est_sorted[left] - ref_t) <= np.abs(est_sorted[right] - ref_t), left, right)
    ok = np.abs(est_sorted[pick] - ref_t) <= MATCH_TOLERANCE_S
    return order[pick], ok


def trapezoid(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(t))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="bag рейса (с GNSS master)")
    parser.add_argument("--result", required=True, help="bag с записью /result/velocity и /result/position")
    defaults = node_defaults()
    parser.add_argument("--mgrs-grid", default=defaults["mgrs_grid"], help="квадрат MGRS, в котором узел выдавал x, y")
    parser.add_argument("--antenna-behind", type=float, default=defaults["base_link_ahead_of_antenna_m"],
                        help="антенна master позади base_link, м")
    parser.add_argument("--antenna-above", type=float, default=defaults["base_link_below_antenna_m"],
                        help="антенна master выше base_link, м")
    args = parser.parse_args(argv)

    source = read_bag(args.input, {"/sensing/gnss/master/vel", "/sensing/gnss/master/fix", "/vehicle/driver_position_cmd"})
    result = read_bag(args.result, {"/result/velocity", "/result/position", "/diagnostics"})

    ref_vel = source.get("/sensing/gnss/master/vel", [])
    ref_t = np.array([stamp_s(m) for _, m in ref_vel])
    ref_v = np.array([math.hypot(m.twist.linear.x, m.twist.linear.y) for _, m in ref_vel])
    est_vel = result.get("/result/velocity", [])
    est_t = np.array([stamp_s(m) for _, m in est_vel])
    est_v = np.array([m.velocity for _, m in est_vel])
    if not len(ref_t) or not len(est_t):
        raise SystemExit("нет /sensing/gnss/master/vel во входном bag или /result/velocity в записи результата")

    print(f"эталон GNSS: {len(ref_t)} точек, {ref_t[-1] - ref_t[0]:.0f} с; оценок скорости: {len(est_t)}")
    duration = est_t[-1] - est_t[0]
    if duration > 0:
        print(f"частота /result/velocity: {(len(est_t) - 1) / duration:.1f} Гц (по меткам)")

    # --- критерий 1: скорость ---
    idx, ok = nearest(ref_t, est_t)
    if not ok.any():
        raise SystemExit("метки /result/velocity не пересекаются по времени с эталоном GNSS (±0,05 с)")
    error = est_v[idx[ok]] - ref_v[ok]
    print("\nСкорость (м/с): RMSE {:.3f}  MAE {:.3f}  bias {:+.3f}   сопоставлено {:.1f} % точек эталона".format(
        math.sqrt(np.mean(error ** 2)), np.mean(np.abs(error)), np.mean(error), 100 * ok.mean()))
    handle = source.get("/vehicle/driver_position_cmd", [])
    if handle:
        handle_t = np.array([stamp_s(m) for _, m in handle])
        handle_u = np.array([m.position for _, m in handle])
        at = np.clip(np.searchsorted(handle_t, ref_t[ok], side="right") - 1, 0, len(handle_u) - 1)
        u = handle_u[at]
        regimes = {"разгон (ручка > 0)": (u > 0) & (ref_v[ok] >= STOP_MPS),
                   "торможение (ручка < 0)": (u < 0) & (ref_v[ok] >= STOP_MPS),
                   "выбег (ручка 0)": (u == 0) & (ref_v[ok] >= STOP_MPS),
                   "стоянка (< 1 км/ч)": ref_v[ok] < STOP_MPS}
        for name, mask in regimes.items():
            if mask.any():
                print(f"  bias, {name}: {np.mean(error[mask]):+.3f} м/с")

    # --- критерий 2: путь вдоль пути ---
    est_s = trapezoid(est_t, est_v)
    ref_s = trapezoid(ref_t, ref_v)
    start = int(np.searchsorted(ref_t, est_t[0]))           # эталон — от первой выданной оценки
    if start < len(ref_s):
        ref_s = ref_s - ref_s[start]
    idx, ok = nearest(ref_t, est_t)
    ok &= np.arange(len(ref_t)) >= start
    if ok.any():
        along = est_s[idx[ok]] - ref_s[ok]
        total = ref_s[ok][-1]
        print("\nПуть вдоль пути (м): MEAN {:.2f}  MAX {:.2f}  RMSE {:.2f}   дрейф в конце {:.3f} % от {:.0f} м".format(
            np.mean(np.abs(along)), np.max(np.abs(along)), math.sqrt(np.mean(along ** 2)),
            100 * abs(along[-1]) / total, total))

    positions = result.get("/result/position", [])
    if positions:
        pos_t = np.array([stamp_s(m) for _, m in positions])
        xyz = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in positions])
        frames = {m.header.frame_id for _, m in positions}
        fixes = source.get("/sensing/gnss/master/fix", [])
        if fixes and frames == {"map"}:
            grid = MgrsGrid.parse(args.mgrs_grid)
            fix_t = np.array([stamp_s(m) for _, m in fixes])
            fix_xyz = np.array([[*grid.to_grid(*utm_forward(m.latitude, m.longitude, grid.zone)), m.altitude]
                                for _, m in fixes])
            good = np.array([m.status.status >= 0 for _, m in fixes])
            yaw = np.array([2 * math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w) for _, m in positions])
            idx, ok = nearest(fix_t, pos_t)
            ok &= good
            if not ok.any():
                print("\nПоложение: метки /result/position не пересекаются по времени с фиксами GNSS")
                return
            # фикс — место антенны; base_link впереди неё по курсу вагона и ниже
            reference = fix_xyz[ok] + np.c_[args.antenna_behind * np.cos(yaw[idx[ok]]),
                                            args.antenna_behind * np.sin(yaw[idx[ok]]),
                                            np.full(ok.sum(), -args.antenna_above)]
            horizontal = np.linalg.norm(xyz[idx[ok], :2] - reference[:, :2], axis=1)
            vertical = xyz[idx[ok], 2] - reference[:, 2]
            print("\nПоложение base_link против фикса GNSS, перенесённого в base_link (м): по горизонтали медиана {:.2f}  среднее {:.2f}  "
                  "RMSE {:.2f}  макс {:.2f};  по высоте среднее {:+.2f}".format(
                      np.median(horizontal), np.mean(horizontal), math.sqrt(np.mean(horizontal ** 2)),
                      np.max(horizontal), np.mean(vertical)))
        else:
            print(f"\nПоложение в кадре {sorted(frames)}: сравнение с GNSS по x, y пропущено (нет выставки)")

    # --- задержка ---
    for _, array in result.get("/diagnostics", [])[-1:]:
        for status in array.status:
            if status.name.endswith("задержка"):
                print("\n" + ", ".join(f"{kv.key}: {kv.value}" for kv in status.values))
    print(f"\n(скорость в км/ч = м/с × {KMH_PER_MPS})")


if __name__ == "__main__":
    main()
