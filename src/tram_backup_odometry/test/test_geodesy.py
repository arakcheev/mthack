import pytest

from tram_backup_odometry.domain.geodesy import MgrsGrid, utm_forward

# Эталон — pyproj (EPSG:4326 → EPSG:32637), посчитан заранее: в образе ROS pyproj нет.
REFERENCE = [
    (55.805, 37.425, 401281.2440, 6185499.3567),
    (55.8, 37.35, 396567.5620, 6185052.4198),
    (55.81, 37.49, 405367.1288, 6185965.0193),
    (54.0, 36.0, 303379.1016, 5987687.7104),
]


@pytest.mark.parametrize("lat, lon, easting, northing", REFERENCE)
def test_utm_matches_pyproj_to_a_millimetre(lat, lon, easting, northing):
    e, n = utm_forward(lat, lon, 37)
    assert e == pytest.approx(easting, abs=1e-3)
    assert n == pytest.approx(northing, abs=1e-3)


@pytest.mark.parametrize("code, easting, northing", [
    ("37UDB", 400_000.0, 6_100_000.0),     # восточная часть маршрута №10
    ("37UCB", 300_000.0, 6_100_000.0),     # западная часть
    ("31UDQ", 400_000.0, 5_400_000.0),     # Париж
])
def test_mgrs_grid_origin(code, easting, northing):
    grid = MgrsGrid.parse(code)
    assert (grid.origin_easting, grid.origin_northing) == (easting, northing)


def test_organizers_example_in_grid_37ucb():
    """Контрольный пример: MGRS 37UCB, широта 55,8088325462547, долгота 37,4602768500852 → x, y эталона."""
    grid = MgrsGrid.parse("37UCB")
    x, y = grid.to_grid(*utm_forward(55.8088325462547, 37.4602768500852, grid.zone))
    assert x == pytest.approx(103501.6309, abs=1e-3)
    assert y == pytest.approx(85876.1201, abs=1e-3)


def test_route_crosses_square_border_and_stays_in_one_grid():
    grid = MgrsGrid.parse("37UDB")
    west_x, _ = grid.to_grid(*utm_forward(55.8, 37.35, 37))
    east_x, _ = grid.to_grid(*utm_forward(55.81, 37.49, 37))
    assert west_x < 0 < east_x < 100_000     # один квадрат на весь маршрут: x выходит за 0


@pytest.mark.parametrize("code", ["37ZDB", "37UIB", "99UDB", "37CDB"])
def test_bad_grid_codes_are_rejected(code):
    with pytest.raises(ValueError):
        MgrsGrid.parse(code)
