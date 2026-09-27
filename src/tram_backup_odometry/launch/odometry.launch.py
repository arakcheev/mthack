"""Запуск резервной одометрии для одного вагона:

    ros2 launch tram_backup_odometry odometry.launch.py vehicle:=30618
    ros2 launch tram_backup_odometry odometry.launch.py vehicle:=30639

Параметры узла — config/params.yaml. Любой из них переопределяется аргументом того же имени, например
mgrs_grid:=37UDB или start_fix_topic:=/sensing/gnss/rover/fix. Значения по умолчанию берутся из того же
params.yaml, тип аргумента — по типу значения в нём.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from tram_backup_odometry.ros.defaults import NODE_NAME, PACKAGE, node_defaults, params_file


def generate_launch_description():
    defaults = node_defaults()
    arguments = [DeclareLaunchArgument(name, default_value=str(value)) for name, value in defaults.items()]
    # тип — как в params.yaml: номер вагона остаётся строкой, иначе launch передал бы его числом
    overrides = {name: ParameterValue(LaunchConfiguration(name), value_type=type(value))
                 for name, value in defaults.items()}
    node = Node(
        package=PACKAGE,
        executable="odometry_node",
        name=NODE_NAME,
        output="screen",
        parameters=[str(params_file()), overrides],
        respawn=True,              # упал — поднять заново (не должно случаться; страховка)
        respawn_delay=1.0,
        additional_env={"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},  # матрицы 3 × 3: потоки не нужны
    )
    return LaunchDescription([*arguments, node])
