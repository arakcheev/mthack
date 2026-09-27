"""Перевод между сообщениями ROS и величинами оценки: единственное место, где заполняются поля выходных
сообщений /result/* (диагностика — ros/diagnostics.py)."""
import math

from builtin_interfaces.msg import Time
from geometry_msgs.msg import AccelStamped
from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import VelocitySensor

from tram_backup_odometry.domain.estimator import Estimate
from tram_backup_odometry.domain.pose import Pose
from tram_backup_odometry.domain.units import NS_PER_S

# Дисперсии углов положения (крен, тангаж, курс), рад². Крен и тангаж не оцениваются; курс — направление линии
# карты, его ошибка — доли градуса на прямых и больше в кривых.
ROTATION_VARIANCE = (0.01, 0.01, 0.01)


def stamp_ns(stamp: Time) -> int:
    return int(stamp.sec) * NS_PER_S + int(stamp.nanosec)


def to_stamp(time_ns: int) -> Time:
    sec, nanosec = divmod(int(time_ns), NS_PER_S)
    return Time(sec=int(sec), nanosec=int(nanosec))


def velocity_message(stamp_ns: int, estimate: Estimate, frame_id: str) -> VelocitySensor:
    msg = VelocitySensor()
    msg.header.stamp = to_stamp(stamp_ns)
    msg.header.frame_id = frame_id
    msg.velocity = float(estimate.speed)
    return msg


def acceleration_message(stamp_ns: int, estimate: Estimate, frame_id: str) -> AccelStamped:
    msg = AccelStamped()
    msg.header.stamp = to_stamp(stamp_ns)
    msg.header.frame_id = frame_id
    msg.accel.linear.x = float(estimate.acceleration)
    return msg


def odometry_message(stamp_ns: int, estimate: Estimate, pose: Pose, frame_id: str, child_frame_id: str) -> Odometry:
    """Положение в pose.pose.position (м), курс в orientation, продольная скорость в twist.twist.linear.x (м/с)."""
    msg = Odometry()
    msg.header.stamp = to_stamp(stamp_ns)
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id
    position = msg.pose.pose.position
    position.x, position.y, position.z = float(pose.x), float(pose.y), float(pose.z)
    orientation = msg.pose.pose.orientation
    orientation.z, orientation.w = math.sin(pose.yaw / 2), math.cos(pose.yaw / 2)
    covariance = [0.0] * 36                       # 6 × 6 построчно: x, y, z, крен, тангаж, курс
    for row in range(3):
        for col in range(3):
            covariance[row * 6 + col] = float(pose.covariance_xyz[row * 3 + col])
    for axis, variance in enumerate(ROTATION_VARIANCE, start=3):
        covariance[axis * 6 + axis] = variance
    msg.pose.covariance = covariance
    msg.twist.twist.linear.x = float(estimate.speed)
    twist_covariance = [0.0] * 36
    twist_covariance[0] = float(estimate.speed_variance)
    msg.twist.covariance = twist_covariance
    return msg
