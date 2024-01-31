import pytest
import numpy as np
from H5CosmoKit import Pk_suffix

def test_gas():
    assert Pk_suffix([0]) == 'g', "Expected 'g' for gas"

def test_cold_dark_matter():
    assert Pk_suffix([1]) == 'c', "Expected 'c' for cold dark matter"

def test_stars():
    assert Pk_suffix([4]) == 's', "Expected 's' for stars"

def test_black_holes():
    assert Pk_suffix([5]) == 'bh', "Expected 'bh' for black holes"

def test_all_combined():
    assert Pk_suffix([0, 1, 4, 5]) == 'm', "Expected 'm' for all combined"

def test_invalid_input_empty():
    with pytest.raises(Exception) as excinfo:
        Pk_suffix([])
    assert "No label found for ptype" in str(excinfo.value), "Expected exception for empty list"

def test_invalid_input_unrecognized():
    with pytest.raises(Exception) as excinfo:
        Pk_suffix([2])
    assert "No label found for ptype" in str(excinfo.value), "Expected exception for unrecognized ptype"

def test_invalid_input_none():
    with pytest.raises(Exception) as excinfo:
        Pk_suffix(None)
    assert "No label found for ptype" in str(excinfo.value), "Expected exception for None input"
