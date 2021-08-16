"""GLMM encoders for high-cardinality categorical features"""

from setuptools import find_packages, setup

EXTRAS_REQUIRE = {
}

INSTALL_REQUIRES = [
    "numpy>=1.19.2",
    "pandas==1.3.1",
    "seaborn==0.11.1",
    "tensorflow==2.6.0",
    "tensorflow-probability==0.13.0",
]

TESTS_REQUIRE = []

setup(
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    python_requires="~=3.7",
    setup_requires=["pytest-runner>=5.1,<6.0"],
    tests_require=TESTS_REQUIRE,
    scripts=[]
)
