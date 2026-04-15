"""Unit tests for Misc.py — no GL or Qt dependencies."""
import pytest
from Misc import depth, FixedSide, FixedSides


# ---------------------------------------------------------------------------
# depth()
# ---------------------------------------------------------------------------

def test_depth_flat_list():
    assert depth([1, 2, 3]) == 1

def test_depth_nested_list():
    assert depth([[1, 2], [3, 4]]) == 2

def test_depth_scalar():
    assert depth(42) == 0

def test_depth_empty_nested():
    assert depth([[]]) == 2


# ---------------------------------------------------------------------------
# FixedSide
# ---------------------------------------------------------------------------

def test_fixedside_stores_ax_and_dir():
    fs = FixedSide(1, 0)
    assert fs.ax == 1
    assert fs.dir == 0

def test_fixedside_unique_returns_true_when_not_in_list():
    fs = FixedSide(0, 1)
    others = [FixedSide(1, 0), FixedSide(2, 1)]
    assert fs.unique(others) is True

def test_fixedside_unique_returns_false_when_duplicate():
    fs = FixedSide(1, 0)
    others = [FixedSide(1, 0), FixedSide(2, 1)]
    assert fs.unique(others) is False

def test_fixedside_unique_nested_list():
    fs = FixedSide(0, 0)
    others = [[FixedSide(0, 0), FixedSide(1, 1)], [FixedSide(2, 0)]]
    assert fs.unique(others) is False

def test_fixedside_unique_nested_list_not_found():
    fs = FixedSide(0, 1)
    others = [[Fixed_Side := FixedSide(1, 0)], [FixedSide(2, 1)]]
    assert fs.unique(others) is True


# ---------------------------------------------------------------------------
# FixedSides — requires a parent-like object with .sax and .rot attributes
# ---------------------------------------------------------------------------

class _FakeParent:
    """Minimal stand-in for the Types object that FixedSides needs."""
    def __init__(self, sax=2):
        self.sax = sax
        self.rot = True


def test_fixedsides_default_has_two_sides():
    parent = _FakeParent(sax=2)
    fs = FixedSides(parent)
    assert len(fs.sides) == 2

def test_fixedsides_default_sides_are_axis2():
    parent = _FakeParent(sax=2)
    fs = FixedSides(parent)
    assert fs.sides[0][0].ax == 2
    assert fs.sides[1][0].ax == 2

def test_fixedsides_from_string_round_trips():
    parent = _FakeParent(sax=0)
    fs = FixedSides(parent, side_str="0,0:1,1")
    assert fs.sides[0][0].ax == 0
    assert fs.sides[0][0].dir == 0
    assert fs.sides[1][0].ax == 1
    assert fs.sides[1][0].dir == 1

def test_fixedsides_rotation_disabled_when_sax_aligned():
    parent = _FakeParent(sax=0)
    # side on axis 0 → rotation should be disabled
    FixedSides(parent, side_str="0,0:1,1")
    assert parent.rot is False

def test_fixedsides_rotation_enabled_when_sax_not_aligned():
    parent = _FakeParent(sax=2)
    FixedSides(parent, side_str="0,0:1,1")
    assert parent.rot is True

def test_fixedsides_unblocked_excludes_fixed():
    parent = _FakeParent(sax=2)
    fs = FixedSides(parent, side_str="2,0:2,1")
    unblocked_pairs = [(s.ax, s.dir) for s in fs.unblocked]
    assert (2, 0) not in unblocked_pairs
    assert (2, 1) not in unblocked_pairs
