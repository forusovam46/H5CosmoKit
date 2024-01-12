def Pk_suffix(ptype):
    """
    Maps a particle type to its corresponding label.
    
    Args:
        ptype (list): Particle type identifier(s), expected values are [0], [1], [4], [5], or [0,1,4,5].
    
    Returns:
        str: Label of the particle type ('g' for gas, 'c' for cold dark matter, 's' for stars, 'bh' for black holes, 'm' for all combined).
    
    Raises:
        Exception: If no label is found for the provided ptype.
    """
    if   ptype == [0]:            return 'g'
    elif ptype == [1]:            return 'c'
    elif ptype == [4]:            return 's'
    elif ptype == [5]:            return 'bh'
    elif ptype == [0, 1, 4, 5]:   return 'm'
    else:   raise Exception('No label found for ptype')

# # Test the function when the script is run directly
# if __name__ == '__main__':
#     test_ptypes = [[0], [1], [4], [5], [0, 1, 4, 5]]
#     for ptype in test_ptypes:
#         print(f"Particle type {ptype} is labeled as '{Pk_suffix(ptype)}'")

import numpy as np

def even_numbers(n):
    """Return a list of all even numbers up to n
    """
    result = []
    a = 2
    while a < n:
        result.append(a)
        a = a+2
    return result

def fibonacci_numbers(n):
    """Return a list of all Fibonnaci numbers up to n
    """
    result = []
    a, b = 1, 1
    while a < n:
        result.append(a)
        a, b = b, a+b
    return result

def prime_numbers(n):
    """Return a list of all prime numbers up to n
    """
    integers = np.linspace(1, n, n, dtype=int)
    seive = np.array([True for i in range(n)])
    for i in range(2, n):
        seive[i+i-1::i] = False
    result = integers[seive]
    result = list(result[1:])
    return result

def get_sequence(name):
    """Return the appropriate sequence function given a string name
    """
    if name=='even':
        return even_numbers
    elif name=='fibonacci':
        return fibonacci_numbers
    elif name=='prime':
        return prime_numbers
    else:
        raise ValueError('name should be one of [even, fibonacci, prime]')

if __name__ == "__main__":
    n = 25
    for sequence_name in ['even', 'fibonacci', 'prime']:
        sequence_func = get_sequence(sequence_name)
        print(f'{sequence_name} numbers below {n}: {sequence_func(n)}')