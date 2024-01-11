from ..H5CosmoKit import get_sequence

def intersection(list1, list2):
    """Return a list containing elements which are both in list1 and list2"""
    result = [value for value in list1 if value in list2]
    return result

def get_intersection_of_sequences(n, seq_name1, seq_name2):
    """Returns a list of numbers below n appearing in the two named sequences"""
    seq_func1 = get_sequence(seq_name1)
    seq1 = seq_func1(n)
    seq_func2 = get_sequence(seq_name2)
    seq2 = seq_func2(n)
    common_elements = intersection(seq1, seq2)
    return common_elements

if __name__ == "__main__":
    n = 50
    fib_primes = get_intersection_of_sequences(n, 'prime', 'fibonacci')
    n_fibonnaci_primes = len(fib_primes)
    print(f'There are {n_fibonnaci_primes} Fibonacci primes below {n}. These are:')
    for p in fib_primes:
        print(f'   {p}')