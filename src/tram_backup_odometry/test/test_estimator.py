"""Оценка на синтетических входах: разгон, сбои датчиков, мусор на входе. Числа модели — из пакета."""
import math

import pytest

from conftest import NS
from tram_backup_odometry.domain.motion import MAX_SPEED_MPS
from tram_backup_odometry.domain.units import KMH_PER_MPS
from tram_backup_odometry.factory import UnknownVehicle, build_estimator

WHEEL_PERIOD_NS = 50_000_000     # тележки шлют 20 Гц
TICK_NS = 20_000_000             # такт 50 Гц
LATENCY_NS = 40_000_000          # сообщения приходят на 40 мс позже метки


def drive(estimator, seconds: float, handle: int, wheel_kmh, t0_ns: int = 0):
    """Кормит оценку показаниями и тактами; wheel_kmh(t_s, bogie) — показание (км/ч) или None — молчит."""
    outputs = []
    wheel_next, tick_next, end = t0_ns, t0_ns, t0_ns + int(seconds * NS)
    estimator.on_handle(t0_ns, handle, t0_ns)
    while tick_next <= end:
        while wheel_next <= tick_next - LATENCY_NS:
            for bogie in ("front", "rear"):
                kmh = wheel_kmh(wheel_next / NS, bogie)
                if kmh is not None:
                    estimator.on_wheel(bogie, wheel_next, kmh, wheel_next + LATENCY_NS)
            estimator.on_handle(wheel_next, handle, wheel_next + LATENCY_NS)
            wheel_next += WHEEL_PERIOD_NS
        output = estimator.tick(tick_next)
        if output is not None:
            outputs.append(output)
        tick_next += TICK_NS
    return outputs


def test_no_wheel_messages_no_output(model):
    estimator = build_estimator(model, "30618")
    assert estimator.tick(NS) is None


def test_unknown_vehicle_is_refused(model):
    with pytest.raises(UnknownVehicle):
        build_estimator(model, "12345")


def test_standing_tram_stays_at_zero(model):
    outputs = drive(build_estimator(model, "30618"), 10.0, 0, lambda t, b: 0.0)
    assert outputs[-1].speed == 0.0
    assert outputs[-1].distance == 0.0


def test_cruising_tram_follows_consistent_wheels(model):
    scale = model.vehicles["30639"].wheel_scale.kmh_per_mps
    outputs = drive(build_estimator(model, "30639"), 20.0, 0, lambda t, b: 10.0 * scale)
    assert outputs[-1].speed == pytest.approx(10.0, abs=0.1)
    assert outputs[-1].distance == pytest.approx(200.0, rel=0.02)


def test_both_bogies_slipping_do_not_drag_the_estimate(model):
    """Обе тележки буксуют: колёса прибавляют 15 км/ч за 1 с. Оценка не должна уйти за колёсами."""
    scale = model.vehicles["30618"].wheel_scale.kmh_per_mps
    true_mps = 8.0

    def wheels(t, bogie):
        slip = 15.0 * min(max(t - 10.0, 0.0), 1.0) if 10.0 <= t < 15.0 else 0.0
        return true_mps * scale + slip

    outputs = drive(build_estimator(model, "30618"), 20.0, 0, wheels)
    during_slip = [o.speed for o in outputs if 11.0 * NS <= o.stamp_ns < 15.0 * NS]
    assert max(during_slip) * KMH_PER_MPS < true_mps * KMH_PER_MPS + 5.0


@pytest.mark.parametrize("bad_kmh", [math.nan, math.inf, -5.0, 1e9])
def test_garbage_readings_are_dropped(model, bad_kmh):
    scale = model.vehicles["30618"].wheel_scale.kmh_per_mps
    estimator = build_estimator(model, "30618")
    outputs = drive(estimator, 10.0, 0, lambda t, b: bad_kmh if 4.0 <= t < 5.0 else 5.0 * scale)
    assert all(math.isfinite(o.speed) and 0.0 <= o.speed <= MAX_SPEED_MPS for o in outputs)
    assert outputs[-1].speed == pytest.approx(5.0, abs=0.2)
    assert estimator.status().rejected_on_input > 0


def test_silent_bogies_continue_by_the_equation(model):
    scale = model.vehicles["30618"].wheel_scale.kmh_per_mps
    outputs = drive(build_estimator(model, "30618"), 20.0, 0, lambda t, b: None if 8.0 <= t < 18.0 else 6.0 * scale)
    stamps = [o.stamp_ns for o in outputs]
    assert all(b >= a for a, b in zip(stamps, stamps[1:]))               # время выхода не идёт назад
    silent = [o for o in outputs if 9.0 * NS <= o.stamp_ns < 17.0 * NS]
    assert silent and all(not o.fresh for o in silent)                    # выход не прерывается
    assert all(o.speed < 6.0 for o in silent)                             # без тяги вагон замедляется


def test_stamps_going_backwards_or_to_zero_do_not_crash(model):
    scale = model.vehicles["30618"].wheel_scale.kmh_per_mps
    estimator = build_estimator(model, "30618")
    drive(estimator, 5.0, 0, lambda t, b: 5.0 * scale)
    for stamp in (0, -NS, 10**18, 3 * NS):
        estimator.on_wheel("front", stamp, 5.0 * scale, 6 * NS)
        estimator.on_handle(stamp, 3, 6 * NS)
        output = estimator.tick(6 * NS)
        assert math.isfinite(output.speed) and math.isfinite(output.distance)


def test_handle_stamp_from_the_far_future_does_not_freeze_the_handle(model):
    scale = model.vehicles["30618"].wheel_scale.kmh_per_mps
    estimator = build_estimator(model, "30618")
    drive(estimator, 5.0, 0, lambda t, b: 5.0 * scale)
    estimator.on_handle(10**18, 0, 5 * NS)                  # сбой метки: 30 лет вперёд
    outputs = drive(estimator, 20.0, 15, lambda t, b: None, t0_ns=5 * NS)   # полная тяга, колёса молчат
    assert outputs[-1].speed > 5.0                          # ручка после сбоя работает: вагон разгоняется


def test_handle_stamp_from_the_future_before_the_first_wheel_message_is_discarded(model):
    scale = model.vehicles["30618"].wheel_scale.kmh_per_mps
    estimator = build_estimator(model, "30618")
    estimator.on_handle(10**18, 0, 0)                       # сбой метки ещё до первых данных колёс
    outputs = drive(estimator, 20.0, 15, lambda t, b: 0.0 if t < 1.0 else None)
    assert outputs[-1].speed > 5.0                          # ручка работает: вагон разгоняется по уравнению


def test_handle_nan_is_dropped(model):
    estimator = build_estimator(model, "30618")
    estimator.on_handle(0, float("nan"), 0)
    assert estimator.status().rejected_on_input == 1
