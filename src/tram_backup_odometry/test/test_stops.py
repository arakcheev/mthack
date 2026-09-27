"""Сброс ошибки пути на остановках: когда вагон ставится на место стоянки, а когда нет."""
import numpy as np

from tram_backup_odometry.domain.stops import StopReset, StopResetConfig
from tram_backup_odometry.domain.track import LinePosition, PlacedTrack

from conftest import NS, straight_line

CONFIG = StopResetConfig(still_speed_mps=0.1, confirm_s=5.0, window_m=10.0, window_per_metre=0.006)


def drive_then_stop(reset: StopReset, line_s: float, since_anchor_m: float, still_s: float) -> list:
    """Вагон едет, затем стоит still_s секунд в месте line_s; такт 0,1 с. Все ответы observe за стоянку."""
    reset.observe(0, 5.0, line_s - 1.0, since_anchor_m)
    return [reset.observe(int(t * NS), 0.0, line_s, since_anchor_m) for t in np.arange(0.1, still_s, 0.1)]


def test_a_stop_near_a_known_place_moves_the_tram_there_once():
    answers = drive_then_stop(StopReset(np.array([1000.0, 2000.0]), CONFIG), 1006.0, 500.0, 20.0)
    moves = [a for a in answers if a is not None]
    assert moves == [1000.0]


def test_a_short_stop_does_not_move_the_tram():
    answers = drive_then_stop(StopReset(np.array([1000.0]), CONFIG), 1006.0, 500.0, 4.0)
    assert all(a is None for a in answers)


def test_a_stop_outside_the_window_does_not_move_the_tram():
    # окно 10 м + 0,6 % от 500 м = 13 м; место стоянки в 20 м
    answers = drive_then_stop(StopReset(np.array([1000.0]), CONFIG), 1020.0, 500.0, 20.0)
    assert all(a is None for a in answers)


def test_the_window_grows_with_the_path_since_the_last_known_place():
    # те же 20 м, но от последнего точно известного места 3 км: окно 10 + 18 = 28 м
    answers = drive_then_stop(StopReset(np.array([1000.0]), CONFIG), 1020.0, 3000.0, 20.0)
    assert [a for a in answers if a is not None] == [1000.0]


def test_standing_at_the_start_does_not_move_the_tram():
    reset = StopReset(np.array([100.0]), CONFIG)
    answers = [reset.observe(int(t * NS), 0.0, 103.0, 0.0) for t in np.arange(0.0, 20.0, 0.1)]
    assert all(a is None for a in answers)


def test_a_line_without_known_places_never_moves_the_tram():
    answers = drive_then_stop(StopReset(np.empty(0), CONFIG), 1006.0, 500.0, 20.0)
    assert all(a is None for a in answers)


def test_after_a_move_the_place_and_the_grade_follow_the_new_place():
    line = straight_line(0, 0.0, 0.0, 0.0, 3000.0)
    line = type(line)(**{**line.__dict__, "grade": np.where(line.s < 1000.0, 0.0, 0.02)})
    position = LinePosition(100.0)
    track = PlacedTrack(line, 100.0, position)
    assert track.grade_and_curvature(903.0)[0] == 0.02          # 100 + 903 = 1003 м: уже подъём
    position.move_to(995.0, 903.0)                               # сброс: вагон на самом деле в 995 м
    assert position.at(903.0) == 995.0
    assert position.at(913.0) == 1005.0
    assert track.grade_and_curvature(903.0)[0] == 0.0           # 995 м: ещё ровно
    assert position.since_anchor_m(913.0) == 10.0
