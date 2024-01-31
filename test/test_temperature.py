import numpy as np
import pytest
from H5CosmoKit import temperature

def test_temperature_basic():
    """
    Basic test with simple values
    """
    U = np.array([1.0, 2.0, 3.0])  
    ne = np.array([0.5, 1.0, 1.5])

    expected_output = np.array([67.29644634, 102.22171256, 123.60427987])  
    np.testing.assert_allclose(temperature(U, ne), expected_output)

def test_temperature_with_zeros():
    """    
    Test with zero values to ensure no division by zero or other errors
    """
    U = np.array([0, 0, 0])
    ne = np.array([0, 0, 0])

    expected_output = np.array([0, 0, 0])
    np.testing.assert_allclose(temperature(U, ne), expected_output)

def test_temperature_negative_values():
    """
    Test with negative values if applicable
    """
    U = np.array([-1, -2, -3])
    ne = np.array([-0.5, -1, -1.5])

    expected_output = np.array([-183.545274, -2693.39313357, 756.97714879])
    np.testing.assert_allclose(temperature(U, ne), expected_output)

def test_temperature_invalid_input():
    """
    Test with invalid input types
    """
    U = "invalid input"
    ne = "invalid input"

    with pytest.raises(TypeError):
        temperature(U, ne)
