from setuptools import find_packages, setup

setup(
    name="final_project",
    packages=find_packages(exclude=["final_project_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
