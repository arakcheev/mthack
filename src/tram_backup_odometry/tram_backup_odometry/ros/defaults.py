"""Значения параметров узла по умолчанию — единственный источник: config/params.yaml пакета.
Узел, launch-файл и evaluate берут числа отсюда, своих копий не держат."""
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory

PACKAGE = "tram_backup_odometry"
NODE_NAME = "tram_backup_odometry"


def share_path(*parts: str) -> Path:
    return Path(get_package_share_directory(PACKAGE)).joinpath(*parts)


def params_file() -> Path:
    return share_path("config", "params.yaml")


def node_defaults() -> dict:
    """{параметр: значение} из config/params.yaml."""
    document = yaml.safe_load(params_file().read_text(encoding="utf-8"))
    return dict(document[NODE_NAME]["ros__parameters"])
