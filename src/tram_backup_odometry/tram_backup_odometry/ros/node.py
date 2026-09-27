"""Узел ROS 2 резервной одометрии трамвая.

Вход:  /vehicle/front_bogie_velocity, /vehicle/rear_bogie_velocity (VelocitySensor, км/ч — так в записях),
       /vehicle/driver_position_cmd (DriverControllerCommand), GNSS master fix — только первые секунды.
Выход: /result/velocity (VelocitySensor, м/с), /result/position (nav_msgs/Odometry), /result/acceleration
       (AccelStamped), /diagnostics (DiagnosticArray): проверка колёс, выставка, задержка и частота.

Метки выхода — время датчиков колёс (header.stamp входа), как у входа. Такты и время прихода считаются по
монотонным часам компьютера: узел работает и с use_sim_time, и без /clock.
"""
import functools
import traceback
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import AccelStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus
from tram_vehicle_msgs.msg import VelocitySensor

try:
    from tram_vehicle_msgs.msg import DriverControllerCommand
except ImportError:  # пакет сообщений собран без типа ручки: узел считает по тележкам, а не падает
    DriverControllerCommand = None

from tram_backup_odometry import model_file
from tram_backup_odometry.app import Output, PoseSettings, TramOdometry
from tram_backup_odometry.domain.geodesy import MgrsGrid
from tram_backup_odometry.domain.pose import AntennaOffset, PoseUncertainty
from tram_backup_odometry.domain.start_fix import GeoFix, StartFixConfig
from tram_backup_odometry.domain.units import KMH_PER_MPS, NS_PER_S
from tram_backup_odometry.factory import build_odometry
from tram_backup_odometry.ros import diagnostics, messages
from tram_backup_odometry.ros.defaults import NODE_NAME, node_defaults, share_path
from tram_backup_odometry.ros.latency import LatencyMeter

FRONT_TOPIC = "/vehicle/front_bogie_velocity"
REAR_TOPIC = "/vehicle/rear_bogie_velocity"
HANDLE_TOPIC = "/vehicle/driver_position_cmd"

# Вход подписан best-effort: такая подписка получает сообщения и от reliable-издателя (ros2 bag play),
# и от best-effort. Очередь с запасом: тележки шлют ~20 Гц, такт — 50 Гц.
INPUT_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
OUTPUT_QOS = 10
ERROR_LOG_PERIOD_S = 5.0     # одна и та же ошибка колбэка пишется в журнал не чаще


def survives_errors(callback):
    """Колбэк узла не роняет процесс: ошибка пишется в журнал, узел продолжает считать со следующего сообщения.
    Упавший узел потерял бы выставку по GNSS и подстройку модели — это хуже, чем пропущенное сообщение."""
    @functools.wraps(callback)
    def guarded(self, *args):
        try:
            return callback(self, *args)
        except Exception:  # noqa: BLE001 — любой сбой обработки одного сообщения
            self.get_logger().error(f"{callback.__name__}: {traceback.format_exc()}",
                                    throttle_duration_sec=ERROR_LOG_PERIOD_S)
    return guarded


class OdometryNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self._defaults = node_defaults()
        self._steady = Clock(clock_type=ClockType.STEADY_TIME)
        self._odometry, self._vehicle = self._build_odometry()
        self._base_frame = self._param("base_frame")

        self._velocity_pub = self.create_publisher(VelocitySensor, "/result/velocity", OUTPUT_QOS)
        self._position_pub = self.create_publisher(Odometry, "/result/position", OUTPUT_QOS)
        self._acceleration_pub = self.create_publisher(AccelStamped, "/result/acceleration", OUTPUT_QOS)
        self._diagnostics_pub = self.create_publisher(DiagnosticArray, "/diagnostics", OUTPUT_QOS)

        self.create_subscription(VelocitySensor, FRONT_TOPIC, lambda m: self._on_wheel("front", m), INPUT_QOS)
        self.create_subscription(VelocitySensor, REAR_TOPIC, lambda m: self._on_wheel("rear", m), INPUT_QOS)
        if DriverControllerCommand is None:
            self.get_logger().error("в пакете tram_vehicle_msgs нет типа DriverControllerCommand: ручка не читается, "
                                    "скорость считается только по тележкам, точность хуже")
        else:
            self.create_subscription(DriverControllerCommand, HANDLE_TOPIC, self._on_handle, INPUT_QOS)
        fix_topic = self._param("start_fix_topic")
        self._fix_sub = self.create_subscription(NavSatFix, fix_topic, self._on_fix, INPUT_QOS)

        self._latency = LatencyMeter()
        self._published = 0
        rate_hz = self._positive("publish_rate_hz")
        self._diagnostics_every_ns = int(NS_PER_S / self._positive("diagnostics_rate_hz"))
        self._log_every_ns = int(self._positive("log_period_s") * NS_PER_S)
        self._next_diagnostics_ns = self._next_log_ns = 0
        self.create_timer(1.0 / rate_hz, self._on_tick, clock=self._steady)
        self.get_logger().info(f"вагон {self._vehicle}, выход {rate_hz:g} Гц, GNSS для выставки: {fix_topic}")

    # --- сборка ---

    def _param(self, name: str, dynamic: bool = False):
        """Параметр узла; значение по умолчанию — из config/params.yaml."""
        descriptor = ParameterDescriptor(dynamic_typing=dynamic)
        return self.declare_parameter(name, self._defaults[name], descriptor).value

    def _positive(self, name: str) -> float:
        """Положительный параметр; ноль или меньше — ошибка в журнал и значение из config/params.yaml."""
        value = float(self._param(name))
        if value > 0:
            return value
        fallback = float(self._defaults[name])
        self.get_logger().error(f"{name} = {value}: нужно больше нуля, беру {fallback}")
        return fallback

    def _build_odometry(self) -> tuple[TramOdometry, str]:
        model = model_file.load(Path(self._param("model_file") or share_path("config", "model.json")))
        # номер вагона из launch может прийти числом; принимаем и число, и строку
        vehicle = str(self._param("vehicle", dynamic=True)).strip()
        if vehicle not in model.vehicles:
            fallback = str(self._defaults["vehicle"])
            self.get_logger().error(f"вагон {vehicle!r} не описан в модели (есть {sorted(model.vehicles)}); "
                                    f"считаю числами вагона {fallback}")
            vehicle = fallback
        pose_settings = PoseSettings(
            grid=MgrsGrid.parse(self._param("mgrs_grid")),
            antenna=AntennaOffset(ahead_m=float(self._param("base_link_ahead_of_antenna_m")),
                                  below_m=float(self._param("base_link_below_antenna_m"))),
            uncertainty=PoseUncertainty(
                along_at_start_m=float(self._param("position_sigma_at_start_m")),
                along_per_metre=float(self._param("position_sigma_per_metre")),
                across_m=float(self._param("position_sigma_across_m")),
                vertical_m=float(self._param("position_sigma_vertical_m"))),
            map_frame=self._param("map_frame"),
            odom_frame=self._param("odom_frame"),
        )
        start_fix = StartFixConfig(collect_s=float(self._param("start_fix_collect_s")),
                                   wait_s=float(self._param("start_fix_wait_s")))
        delay_s = float(self._param("wheel_reading_delay_s"))
        return build_odometry(model, vehicle, pose_settings, start_fix, delay_s), vehicle

    # --- вход ---

    def _now_ns(self) -> int:
        return self._steady.now().nanoseconds

    @survives_errors
    def _on_wheel(self, bogie: str, msg: VelocitySensor) -> None:
        now = self._now_ns()
        self._latency.received(now)
        self._odometry.on_wheel(bogie, messages.stamp_ns(msg.header.stamp), float(msg.velocity), now)

    @survives_errors
    def _on_handle(self, msg) -> None:
        now = self._now_ns()
        self._latency.received(now)
        self._odometry.on_handle(messages.stamp_ns(msg.header.stamp), int(msg.position), now)

    @survives_errors
    def _on_fix(self, msg: NavSatFix) -> None:
        if msg.status.status < NavSatStatus.STATUS_FIX:
            return
        self._odometry.on_fix(GeoFix(float(msg.latitude), float(msg.longitude)), self._now_ns())

    # --- такт ---

    @survives_errors
    def _on_tick(self) -> None:
        now = self._now_ns()
        output = self._odometry.tick(now)
        if self._fix_sub is not None and not self._odometry.wants_gnss:
            self.destroy_subscription(self._fix_sub)   # GNSS нужен только для выставки на старте
            self._fix_sub = None
            self._log_alignment()
        if output is not None:
            stamp, estimate = output.stamp_ns, output.estimate
            self._velocity_pub.publish(messages.velocity_message(stamp, estimate, self._base_frame))
            self._acceleration_pub.publish(messages.acceleration_message(stamp, estimate, self._base_frame))
            if output.pose is not None:
                self._position_pub.publish(
                    messages.odometry_message(stamp, estimate, output.pose, output.frame_id, self._base_frame))
            self._published += 1
            self._latency.published(self._now_ns())
        if now >= self._next_diagnostics_ns:
            self._next_diagnostics_ns = now + self._diagnostics_every_ns
            self._publish_diagnostics()
        if output is not None and now >= self._next_log_ns:
            self._next_log_ns = now + self._log_every_ns
            self._log_progress(output)

    # --- диагностика и журнал ---

    def _log_alignment(self) -> None:
        a = self._odometry.alignment_status()
        where = (f"направление {a.direction}, фикс в {a.offset_m:.1f} м от линии" if a.direction is not None
                 else "места нет — положение как путь от старта")
        self.get_logger().info(f"выставка: {a.state.value}; {where}; GNSS больше не слушаю")

    def _log_progress(self, output: Output) -> None:
        estimate, status, latency = output.estimate, self._odometry.status(), self._latency.since_start()
        self.get_logger().info(
            f"v={estimate.speed * KMH_PER_MPS:5.1f} км/ч, путь {estimate.distance:7.1f} м, "
            f"тележки [{diagnostics.bogie_summary(status)}], откатов {status.rollbacks}, "
            f"задержка ср {latency.mean_ms:.1f} / макс {latency.max_ms:.1f} мс")

    def _publish_diagnostics(self) -> None:
        statuses = [diagnostics.wheels(self._odometry.status()),
                    diagnostics.alignment(self._odometry.alignment_status()),
                    diagnostics.timing(self._vehicle, self._published, self._latency.take_period(),
                                       self._latency.since_start())]
        self._diagnostics_pub.publish(diagnostics.array(self.get_clock().now().to_msg(), statuses))


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):   # Ctrl-C или остановка launch — штатный выход
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
