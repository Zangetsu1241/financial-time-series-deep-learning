"""Leakage-aware return forecasting research benchmark."""
from .data import synthetic_prices, load_prices, prepare_data
from .benchmark import run_benchmark
__all__ = ["synthetic_prices", "load_prices", "prepare_data", "run_benchmark"]
