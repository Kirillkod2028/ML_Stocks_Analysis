#!/usr/bin/env python3
"""Stable command-line entry point."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_module("ml_stocks_analysis.training_processed", run_name="__main__")
