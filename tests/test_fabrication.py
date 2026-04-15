"""Unit tests for Fabrication.py — no GL or Qt dependencies."""
import math
import numpy as np
import pytest
from Fabrication import angle_between


# ---------------------------------------------------------------------------
# angle_between()
# ---------------------------------------------------------------------------

def test_angle_between_parallel_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert angle_between(v1, v2) == pytest.approx(0.0)

def test_angle_between_antiparallel_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert angle_between(v1, v2) == pytest.approx(math.pi)

def test_angle_between_perpendicular_vectors():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert angle_between(v1, v2) == pytest.approx(math.pi / 2)

def test_angle_between_45_degrees():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    assert angle_between(v1, v2) == pytest.approx(math.pi / 4)

def test_angle_between_non_unit_vectors():
    # Should work regardless of vector magnitudes
    v1 = np.array([2.0, 0.0, 0.0])
    v2 = np.array([0.0, 5.0, 0.0])
    assert angle_between(v1, v2) == pytest.approx(math.pi / 2)
