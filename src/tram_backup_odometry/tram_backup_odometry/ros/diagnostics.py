"""Сообщение /diagnostics: состояние проверки колёс, выставки по GNSS и задержки выхода."""
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from tram_backup_odometry.app import AlignmentStatus
from tram_backup_odometry.domain.estimator import Status
from tram_backup_odometry.ros.latency import LatencyStats

NAME = "tram_backup_odometry"


def _status(name: str, level: bytes, message: str, values: dict) -> DiagnosticStatus:
    status = DiagnosticStatus(name=f"{NAME}: {name}", level=level, message=message, hardware_id="tram")
    status.values = [KeyValue(key=str(k), value=str(v)) for k, v in values.items()]
    return status


def bogie_summary(status: Status) -> str:
    return ", ".join(f"{bogie}: {fault.value if fault else 'норма'}" for bogie, fault in sorted(status.bogie_faults.items()))


def wheels(status: Status) -> DiagnosticStatus:
    """OK — колёса принимаются; WARN — одна тележка отброшена; ERROR — все, скорость только по уравнению."""
    rejected = [bogie for bogie, fault in status.bogie_faults.items() if fault is not None]
    if status.bogie_faults and len(rejected) == len(status.bogie_faults):
        level, text = DiagnosticStatus.ERROR, "все тележки отброшены: скорость по уравнению движения"
    elif rejected:
        level, text = DiagnosticStatus.WARN, "отброшена тележка: " + ", ".join(sorted(rejected))
    else:
        level, text = DiagnosticStatus.OK, "колёса в норме"
    values = {f"тележка {bogie}": (fault.value if fault else "норма") for bogie, fault in sorted(status.bogie_faults.items())}
    values.update({
        "η тяги": f"{status.eta_traction:.3f}",
        "η торможения": f"{status.eta_brake:.3f}",
        "откатов к уравнению": status.rollbacks,
        "отброшено на входе": status.rejected_on_input,
        **{f"событие: {reason}": count for reason, count in sorted(status.events.items())},
    })
    return _status("колёса", level, text, values)


def alignment(state: AlignmentStatus) -> DiagnosticStatus:
    values = {"направление": state.direction,
              "фикс от линии, м": None if state.offset_m is None else f"{state.offset_m:.2f}"}
    return _status("выставка", DiagnosticStatus.OK, state.state.value, values)


def timing(vehicle: str, published: int, period: LatencyStats, since_start: LatencyStats) -> DiagnosticStatus:
    values = {"вагон": vehicle, "выдано оценок": published, "оценок за период": period.count,
              "задержка за период: средняя, мс": f"{period.mean_ms:.1f}",
              "задержка за период: наибольшая, мс": f"{period.max_ms:.1f}",
              "задержка с начала: средняя, мс": f"{since_start.mean_ms:.1f}",
              "задержка с начала: наибольшая, мс": f"{since_start.max_ms:.1f}"}
    return _status("задержка", DiagnosticStatus.OK, "вход → публикация", values)


def array(stamp, statuses: list) -> DiagnosticArray:
    msg = DiagnosticArray()
    msg.header.stamp = stamp
    msg.status = statuses
    return msg
