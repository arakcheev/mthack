from conftest import NS
from tram_backup_odometry.domain.timeline import MAX_STAMP_LEAD_NS, HandleHistory, SensorClock


def test_stamp_jumping_ahead_is_capped_by_arrival_time():
    clock = SensorClock()
    t0 = clock.start(10 * NS, 10 * NS + 50_000_000)
    jumped = clock.measurement_time(t0, 11 * NS, 10 * NS + 100_000_000)   # метка на 1 с вперёд, пришло через 50 мс
    assert jumped <= t0 + 50_000_000 + MAX_STAMP_LEAD_NS


def test_sensor_time_follows_arrival_clock_between_readings():
    clock = SensorClock()
    clock.start(5 * NS, 7 * NS)
    assert clock.sensor_time(8 * NS) == 6 * NS


def test_handle_ignores_messages_older_than_the_last_one():
    handle = HandleHistory()
    handle.add(2 * NS, 5)
    handle.add(1 * NS, -3)         # метка назад: в историю не попадает, но это последнее полученное
    assert handle.at(3 * NS) == 5.0
    assert handle.latest == -3.0
    assert handle.at(1 * NS) == 0.0


def test_forgetting_old_handle_keeps_answers_from_the_cut_on():
    handle = HandleHistory()
    for i in range(1000):
        handle.add(i * 10_000_000, i % 7)
    before = [handle.at(t) for t in range(5 * NS, 10 * NS, 3_333_333)]
    handle.forget_before(5 * NS)
    after = [handle.at(t) for t in range(5 * NS, 10 * NS, 3_333_333)]
    assert before == after
    assert len(handle) < 1000
