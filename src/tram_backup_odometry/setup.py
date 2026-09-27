from glob import glob

from setuptools import find_packages, setup

PACKAGE = "tram_backup_odometry"

setup(
    name=PACKAGE,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE]),
        ("share/" + PACKAGE, ["package.xml"]),
        ("share/" + PACKAGE + "/config", glob("config/*")),
        ("share/" + PACKAGE + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tram odometry team",
    description="Резервная одометрия трамвая по ручке контроллера и скоростям тележек",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "odometry_node = tram_backup_odometry.ros.node:main",
        "evaluate = tram_backup_odometry.ros.evaluate:main",
    ]},
)
