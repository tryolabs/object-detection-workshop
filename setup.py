from setuptools import setup, find_packages

setup(
    name='workshop',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'click',
        'ipython',
        'ipdb',
        'jupyter',
        'matplotlib',
        'Pillow',
    ],
)
