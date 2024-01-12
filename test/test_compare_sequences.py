import sys
sys.path.append(r'C:\Users\Magdaléna\OneDrive\H5CosmoKit')

from H5CosmoKit.analysis.submodule import get_intersection_of_sequences

from H5CosmoKit.analysis import submodule

def test_compare_fib_primes():
    n = 50
    fib_primes = get_intersection_of_sequences(n, 'prime', 'fibonacci')
    assert fib_primes == [2,3,5,13]