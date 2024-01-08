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

# Test the function when the script is run directly
if __name__ == '__main__':
    test_ptypes = [[0], [1], [4], [5], [0, 1, 4, 5]]
    for ptype in test_ptypes:
        print(f"Particle type {ptype} is labeled as '{Pk_suffix(ptype)}'")
