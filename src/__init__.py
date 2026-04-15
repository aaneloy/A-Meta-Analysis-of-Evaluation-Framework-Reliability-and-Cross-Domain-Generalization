"""
RAG Evaluation Meta-Analysis
============================

A comprehensive comparison of RAG evaluation frameworks.

This package provides:
- Data loading utilities for RAGBench
- Implementations of 9 evaluation frameworks
- Statistical analysis tools
- Visualization utilities

Author: Asif Ahmed Neloy
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Asif Ahmed Neloy"

from .data_loader import DataLoader
from .metrics import MetricsAnalyzer
from .visualization import Visualizer

__all__ = ["DataLoader", "MetricsAnalyzer", "Visualizer"]
