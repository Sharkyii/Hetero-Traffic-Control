"""
Experiments Package: Comparative evaluations, Monte Carlo benchmarks, and publication figures.
"""

from experiments.evaluate_flight import simulate_model
from experiments.generate_figures import generate_all_figures

__all__ = [
    "simulate_model",
    "generate_all_figures",
]
