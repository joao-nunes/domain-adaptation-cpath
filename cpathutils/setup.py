from setuptools import find_packages, setup

setup(
    name='cpathutils',
    packages=['cpathutils',],
    version='0.1.2',
    description='Pathology Datasets',
    author='Joao Nunes',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
