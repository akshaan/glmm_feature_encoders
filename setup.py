"""GLMM encoders for high-cardinality categorical features"""

from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
    "examples": [
        "numpy>=1.19.2",
        "pandas==1.3.1",
        "seaborn==0.11.1",
        "openml==0.12.2",
        "scikit-learn==0.24.2",
        "xgboost==1.4.2",
        "rpy2==3.4.5"
    ]
}

INSTALL_REQUIRES = [
    "tensorflow==2.6.0",
    "tensorflow-probability==0.13.0",
]

TESTS_REQUIRE = [
    "mypy==0.780",
    "pylint>=2.3.1,<2.7.3",
    "pytest>=5.1.1,<6.00",
    "pytest-mock>=3.5.1,<3.6.0",
    "pytest-docstyle>=1.5.0,<2.0.0",
    "pytest-pylint>=0.14.1,<0.15.0",
    "pytest-mypy>=0.4.0,<0.5.0",
]

setup(
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires="~=3.7",
    setup_requires=["pytest-runner>=5.1,<6.0"],
    tests_require=TESTS_REQUIRE,
    scripts=[],
    package_data={"examples": ["r_scripts/*.R"]},
)
