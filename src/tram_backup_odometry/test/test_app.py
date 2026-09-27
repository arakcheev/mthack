"""Одометрия целиком: когда появляется положение и в каком кадре."""
from conftest import NS
from tram_backup_odometry.app import PoseSettings
from tram_backup_odometry.domain.geodesy import MgrsGrid, utm_forward
from tram_backup_odometry.domain.pose import AntennaOffset, PoseUncertainty
from tram_backup_odometry.domain.start_fix import GeoFix, StartFixConfig
from tram_backup_odometry.factory import build_odometry

SETTINGS = PoseSettings(MgrsGrid.parse("37UCB"), AntennaOffset(0.0, 0.0), PoseUncertainty(2.0, 0.003, 1.0, 2.0),
                        "map", "odom")
START = StartFixConfig(collect_s=2.0, wait_s=3.0)


def run(odometry, seconds: float, fix: GeoFix | None):
    outputs = []
    for i in range(int(seconds * 50)):
        now = i * 20_000_000
        if i % 2 == 0:
            odometry.on_handle(now, 0, now)
            odometry.on_wheel("front", now, 0.0, now)
            odometry.on_wheel("rear", now, 0.0, now)
        if fix is not None and i % 5 == 0:
            odometry.on_fix(fix, now)
        outputs.append(odometry.tick(now))
    return [o for o in outputs if o is not None]


def test_with_gnss_at_start_the_pose_is_on_the_map(model):
    line = model.lines[0]
    east, north = line.easting[80], line.northing[80]      # стоим на линии направления 0 у конечной
    lat, lon = _inverse(east, north)
    odometry = build_odometry(model, "30618", SETTINGS, START)
    outputs = run(odometry, 4.0, GeoFix(lat, lon))
    assert outputs[-1].frame_id == "map"
    pose = outputs[-1].pose
    assert abs(pose.x - (east - 300_000.0)) < 3.0 and abs(pose.y - (north - 6_100_000.0)) < 3.0
    assert not odometry.wants_gnss


def test_without_gnss_the_pose_is_relative_after_the_wait(model):
    odometry = build_odometry(model, "30618", SETTINGS, START)
    outputs = run(odometry, 4.0, None)
    early = [o for o in outputs if o.estimate.stamp_ns < 2 * NS]
    assert early and all(o.pose is None for o in early)       # первые секунды ждём GNSS: положения нет
    assert outputs[-1].frame_id == "odom" and outputs[-1].pose.x == 0.0


def _inverse(east: float, north: float) -> tuple[float, float]:
    """Широта и долгота точки UTM 37 перебором (для теста хватает)."""
    lat, lon = 55.8, 37.4
    for _ in range(50):
        e, n = utm_forward(lat, lon, 37)
        lat += (north - n) / 111_320.0
        lon += (east - e) / (111_320.0 * 0.5625)
    return lat, lon


def test_waiting_for_gnss_starts_with_the_first_wheel_message_not_with_the_node(model):
    odometry = build_odometry(model, "30618", SETTINGS, START)
    for i in range(500):                        # узел работает 10 с без данных
        assert odometry.tick(i * 20_000_000) is None
    assert odometry.wants_gnss


def test_output_stamp_is_wheel_stamp_plus_reading_delay(model):
    odometry = build_odometry(model, "30618", SETTINGS, START, wheel_reading_delay_s=0.1)
    outputs = run(odometry, 1.0, None)
    assert all(o.stamp_ns == o.estimate.stamp_ns + 100_000_000 for o in outputs)
