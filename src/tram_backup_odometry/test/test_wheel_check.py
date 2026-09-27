"""Правила проверки колёс по отдельности."""
from conftest import NS
from tram_backup_odometry.domain.wheel_check.checker import WheelChecker
from tram_backup_odometry.domain.wheel_check.types import BogieFault, Prediction, WheelReading


def prediction(speed: float, handle: float = 0.0, shadow: float | None = None) -> Prediction:
    return Prediction(speed=speed, variance=0.01, measurement_variance=0.001,
                      shadow_speed=speed if shadow is None else shadow, shadow_age_s=1.0,
                      handle=handle, model_acceleration=0.0)


def feed(checker, bogie: str, t_s: float, speed: float, predicted: float, handle: float = 0.0):
    return checker.judge(WheelReading(bogie, int(t_s * NS), speed), prediction(predicted, handle))


def test_single_outlier_is_skipped_without_rejecting_the_bogie(model):
    checker = WheelChecker(model.wheel_check)
    for i in range(20):
        feed(checker, "front", i * 0.05, 10.0, 10.0)
    verdict = feed(checker, "front", 1.0, 30.0, 10.0)
    assert not verdict.accept and verdict.reason is None
    assert feed(checker, "front", 1.05, 10.0, 10.0).accept
    assert checker.rejected_bogies() == set()


def test_frozen_sensor_is_rejected_while_the_model_moves(model):
    checker = WheelChecker(model.wheel_check)
    verdicts = [feed(checker, "front", i * 0.05, 8.0, 8.0 - 0.3 * i * 0.05) for i in range(60)]
    assert any(v.reason is BogieFault.FROZEN for v in verdicts)
    assert "front" in checker.rejected_bogies()


def test_bogie_disagreement_under_traction_rejects_the_faster_one(model):
    checker = WheelChecker(model.wheel_check)
    for i in range(10):
        feed(checker, "front", i * 0.05, 10.0, 10.0, handle=5)
        feed(checker, "rear", i * 0.05 + 0.01, 10.0, 10.0, handle=5)
    feed(checker, "front", 0.5, 11.5, 10.0, handle=5)     # передняя ушла вверх при тяге
    feed(checker, "rear", 0.51, 10.0, 10.0, handle=5)
    assert checker.rejected_bogies() == {"front"}


def test_rejected_bogie_returns_after_holding_near_the_prediction(model):
    checker = WheelChecker(model.wheel_check)
    for i in range(10):
        feed(checker, "front", i * 0.05, 10.0, 10.0)
    for i in range(10, 40):                                 # выбросы подряд дольше outlier_run_s
        feed(checker, "front", i * 0.05, 25.0, 10.0)
    assert checker.rejected_bogies() == {"front"}
    returned = [feed(checker, "front", 2.0 + i * 0.05, 10.0, 10.0) for i in range(20)]
    assert any(v.accept for v in returned)
    assert checker.rejected_bogies() == set()
