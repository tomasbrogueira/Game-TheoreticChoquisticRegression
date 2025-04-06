from setuptools import setup, find_packages

setup(
    name="pic_i",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "openpyxl>=3.0.0",
        "tqdm>=4.60.0",
    ],
)
