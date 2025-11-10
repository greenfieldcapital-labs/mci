# MCI - Time Series Simulation with Dynamic Time Warping

A Python package for time series simulation using Dynamic Time Warping (DTW) to identify and leverage similar historical patterns.

## Try it Now in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/greenfieldcapital-labs/mci/blob/release-v1/example.ipynb)

**Click the badge above to run the interactive example in your browser - no installation required!**

The notebook includes:
- Quick demo with synthetic data
- Hyperparameter optimization with Optuna
- Real-world Bitcoin price simulation example

## Quick Start

The method identifies historical patterns similar to current data using DTW, then generates simulations by averaging outcomes from the best matches.

Key features:
- Expanding window analysis for time series
- Automated hyperparameter tuning
- Interactive visualizations with Plotly
- Works with any numeric time series data

## Installation

```bash
pip install git+https://github.com/greenfieldcapital-labs/mci.git
```

## License

See [LICENSE](LICENSE) file for details.
