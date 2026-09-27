import pytest

from conftest import NS, straight_line
from tram_backup_odometry.domain.geodesy import utm_forward
from tram_backup_odometry.domain.start_fix import AlignmentState, GeoFix, StartAlignment, StartFixConfig
from tram_backup_odometry.domain.track import TrackMap

LAT, LON = 55.805, 37.425
CONFIG = StartFixConfig(collect_s=2.0, wait_s=3.0)


def alignment_around_start() -> StartAlignment:
    east, north = utm_forward(LAT, LON, 37)
    line = straight_line(0, east - 100.0, north, 0.0, 5000.0)     # вагон стоит в 100 м от начала линии
    return StartAlignment(TrackMap({0: line}, max_offset_m=30.0), 37, CONFIG)


def test_first_good_fix_places_the_tram_and_later_fixes_refine_it():
    alignment = alignment_around_start()
    alignment.poll(0.0, 0)                        # первые данные колёс
    place = alignment.on_fix(GeoFix(LAT, LON), travelled_m=0.0, now_ns=0)
    assert place.route_s_at_zero == pytest.approx(100.0, abs=2.0)
    assert alignment.state is AlignmentState.REFINING
    alignment.poll(0.0, 1 * NS)
    assert not alignment.finished
    alignment.poll(0.0, 2 * NS)
    assert alignment.state is AlignmentState.ALIGNED
    assert alignment.on_fix(GeoFix(LAT, LON), 0.0, 3 * NS) is None     # после выставки GNSS не принимается


def test_fix_far_from_the_route_is_ignored():
    alignment = alignment_around_start()
    assert alignment.on_fix(GeoFix(LAT + 0.01, LON), 0.0, 0) is None
    assert alignment.state is AlignmentState.WAITING


def test_no_fix_means_relative_odometry_after_the_wait():
    alignment = alignment_around_start()
    alignment.poll(0.0, 0)
    alignment.poll(0.0, 3 * NS)
    assert alignment.state is AlignmentState.RELATIVE


def test_moving_tram_locks_the_place_and_accounts_for_the_path_driven():
    alignment = alignment_around_start()
    alignment.poll(0.0, 0)
    place = alignment.on_fix(GeoFix(LAT, LON), travelled_m=40.0, now_ns=0)
    assert place.route_s_at_zero == pytest.approx(60.0, abs=2.0)
    alignment.poll(45.0, NS // 2)
    assert alignment.state is AlignmentState.ALIGNED


def test_nan_fix_is_ignored():
    alignment = alignment_around_start()
    assert alignment.on_fix(GeoFix(float("nan"), LON), 0.0, 0) is None


def test_fixes_long_before_wheel_data_keep_refining_after_the_wheels_start():
    alignment = alignment_around_start()
    alignment.on_fix(GeoFix(LAT, LON), 0.0, 0)
    alignment.poll(0.0, 60 * NS)                  # колёса пошли через минуту после первого фикса
    assert alignment.state is AlignmentState.REFINING
    alignment.poll(0.0, 62 * NS)
    assert alignment.state is AlignmentState.ALIGNED
