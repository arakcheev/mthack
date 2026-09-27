import math

import pytest

from conftest import straight_line
from tram_backup_odometry.domain.track import PlacedTrack, TrackMap


def test_point_interpolates_and_holds_the_end_of_the_map():
    line = straight_line(0, 1000.0, 2000.0, 90.0, 100.0)    # на север
    east, north, alt, heading = line.point(51.0)
    assert (east, north) == pytest.approx((1000.0, 2051.0))
    assert alt == pytest.approx(150.51)
    assert heading == pytest.approx(math.pi / 2)
    east, north, _, _ = line.point(130.0)                    # за концом карты — её последняя точка
    assert north == pytest.approx(2100.0)


def test_placed_track_reads_grade_at_start_plus_travelled():
    line = straight_line(0, 0.0, 0.0, 0.0, 100.0)
    grade, curvature = PlacedTrack(line, 10.0).grade_and_curvature(20.0)
    assert (grade, curvature) == (pytest.approx(0.01), 0.0)


def _two_parallel_tracks():
    """Направление 0 едет на восток, 1 — на запад по соседнему пути в 4 м севернее."""
    east = straight_line(0, 0.0, 0.0, 0.0, 5000.0)
    west = straight_line(1, 5000.0, 4.0, 180.0, 5000.0)
    return TrackMap({0: east, 1: west}, max_offset_m=30.0)


def test_at_a_terminal_the_direction_with_track_ahead_wins_even_if_farther():
    track_map = _two_parallel_tracks()
    placement = track_map.locate(2.0, 3.0)     # западная конечная, ближе к линии направления 1
    assert placement.line.direction == 0
    assert placement.start_s == pytest.approx(2.0)


def test_in_the_middle_the_nearest_track_wins():
    track_map = _two_parallel_tracks()
    assert track_map.locate(2500.0, 3.5).line.direction == 1
    assert track_map.locate(2500.0, 0.5).line.direction == 0


def test_far_from_the_route_there_is_no_place():
    assert _two_parallel_tracks().locate(2500.0, 100.0) is None
