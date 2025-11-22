from setuptools import setup, find_packages

# Read dependencies from pyproject.toml for compatibility
setup(
    name="mci",
    version="1.2",
    description="Time series forecasting using Dynamic Time Warping",
    packages=find_packages(),
    python_requires=">=3.9",  # Colab uses Python 3.10, relaxed from 3.13
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "fastdtw",
        "plotly",
        "tqdm",
        "optuna",
        "scikit-learn",
        "ipywidgets",
    ],
)
