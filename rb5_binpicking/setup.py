from setuptools import setup, find_packages
import os
from glob import glob

package_name = "rb5_binpicking"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.srdf")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/scripts", glob("scripts/*.py")),
    ],
    scripts=[
        "scripts/depth_to_pointcloud.py",
        "scripts/moveit_pick_place.py",
        "scripts/scene_setup.py",
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hh",
    maintainer_email="gudgh1630@gmail.com",
    description="RB5-850E bin picking — Isaac Sim + Isaac Lab",
    license="Apache-2.0",
    entry_points={},
)
