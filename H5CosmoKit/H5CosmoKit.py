import h5py
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
from scipy.interpolate import NearestNDInterpolator
import sys

###############################################################################################

def download_file(url, local_filename):
    """
    Downloads a file from a specified URL and saves it to a local path.

    This function retrieves a file from the given URL and writes it to a local file, 
    handling it in chunks to manage memory efficiently.

    Args:
        url (str): The URL of the file to download.
        local_filename (str): The local path (including filename) where the file 
                              will be saved.

    Returns:
        str: The path to the downloaded file.

    Raises:
        HTTPError: An error occurs from the HTTP request like 404, 500, etc.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def interpolate_quantity(pos_g, quantity_g, boxSize):
    """
    Interpolates snapshot quantity onto a grid.

    Args:
        pos_g (numpy.ndarray): Particle coordinates in Mpch^{-1}.
        quantity_g (numpy.ndarray): Particle quantity to be interpolated.
        boxSize (float): Box size in Mpch^{-1}.

    Returns:
        scipy.interpolate.NearestNDInterpolator: Interpolation function.
    """
    pos = np.fmod(pos_g, boxSize)
    interp = NearestNDInterpolator(pos, quantity_g, tree_options={'boxsize': boxSize})
    return interp

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

def plot_snapshot(ax, grid_quantity, boxSize, title, quantity):
    """
    Plots a snapshot quantity grid.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
        grid_quantity (numpy.ndarray): Grid of snapshot quantity.
        boxSize (float): Box size in Mpch^{-1}.
        title (str): Title for the plot.
        quantity (str): Quantity name (e.g., 'rho_g' or 'temperature').

    Returns:
        matplotlib.image.AxesImage: The plotted image.
    """

    cmap = plt.cm.get_cmap('Spectral').reversed() if quantity == 'rho_g' else plt.cm.get_cmap('hot')

    im = ax.imshow(grid_quantity[:, :, 128], extent=[0, boxSize, 0, boxSize], origin='lower', norm=LogNorm(), cmap=cmap)
    ax.axis('equal')
    ax.set_xlabel(r'$x \:[Mpc/h$]')
    ax.set_ylabel(r'$y \:[Mpc/h$]')
    ax.set_title(title)
    return im

def read_snapshot_CAMELS(snapshot_path):
    """
    Reads snapshot quantity from a CAMELS quantityset HDF5 file.

    Args:
        snapshot_path (str): The path to the HDF5 snapshot file.

    Returns:
        tuple: A tuple containing:
            - boxSize (float): Box size in Mpch^{-1}.
            - redshift (float): Redshift of the snapshot.
            - pos_g (numpy.ndarray): Particle coordinates in Mpch^{-1}.
            - rho_g (numpy.ndarray): Particle density in (Msun/h)/(ckpc/h)^3.
            - U (numpy.ndarray): Particle internal energy in (km/s)^2.
            - ne (numpy.ndarray): Electron abundance.
    """
    with h5py.File(snapshot_path, 'r') as f:
        boxSize = f['Header'].attrs[u'BoxSize'] / 1e3
        redshift = f['Header'].attrs[u'Redshift']
        pos_g = f['PartType0/Coordinates'][:] / 1e3 
        pos_g = pos_g.astype(np.float32)
        rho_g = f['PartType0/Density'][:] * 1e10
        U = f['PartType0/InternalEnergy'][:]
        ne = f['PartType0/ElectronAbundance'][:]
        np.trim_zeros(rho_g)
    return boxSize, redshift, pos_g, rho_g, U, ne

def preview(path, snapshot_numbers, quantity):
    """
    Creates a plot of snapshot quantity from multiple snapshot files.

    Args:
        path (str): The base path containing the snapshot files.
        snapshot_numbers (list of int): List of snapshot numbers to plot.
        quantity (str): Quantity name (e.g., 'rho_g' or 'temperature').
    """
    snapshot_paths = [os.path.join(path, f'snap_{number:03}.hdf5') for number in snapshot_numbers]
    fig, axes = plt.subplots(1, len(snapshot_numbers), figsize=(7 * len(snapshot_numbers), 5))
    if len(snapshot_numbers) == 1:
        axes = [axes]
    fig.suptitle(path)

    first_time = True
    for snapshot_path, ax in zip(snapshot_paths, axes):
        boxSize, redshift, pos_g, rho_g, U, ne = read_snapshot_CAMELS(snapshot_path)

        if quantity == 'temperature':
            T = temperature(U, ne)  # Calculate temperature
            quantity_g = T
            colorbar_label = "T [K]"  # Label for temperature
        else:  # Assuming 'rho_g' or any other quantity uses rho_g
            quantity_g = rho_g
            colorbar_label = r'$\rho_g \: [\mathrm{M_\odot/h/(ckpc/h)^3}]$'  # Label for density

        interp_quantity = interpolate_quantity(pos_g, quantity_g, boxSize)
        xx = np.linspace(0, boxSize, 256, endpoint=False)
        grid_x, grid_y, grid_z = np.meshgrid(xx, xx, xx)
        grid_quantity = interp_quantity((grid_x, grid_y, grid_z))
        title = f'z={round(redshift)}'
        im = plot_snapshot(ax, grid_quantity, boxSize, title, quantity)

        if first_time:
            f = im
            first_time = False
        cbar = fig.colorbar(f, ax=ax, label=colorbar_label)

    save_plot(fig, path, quantity)

def save_plot(fig, path, quantity):
    """
    Saves the figure to a file with an appropriate filename.

    Args:
        fig (matplotlib.figure.Figure): The figure to be saved.
        path (str): The base path where the plot will be saved.
        quantity (str): Quantity name (e.g., 'rho_g' or 'temperature').
    """
    path_last_parts = os.path.normpath(path).split(os.path.sep)[-2:]
    filename = "_".join(path_last_parts) + f'_{quantity}.png'
    plots_dir = os.path.join(path, 'plots')
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    file_path = os.path.join(plots_dir, filename)
    fig.savefig(file_path)
    plt.show()

def temperature(U, ne):
    """
    Computes the temperature of particles in a snapshot.

    This function calculates the temperature of particles in a snapshot based on their internal energy,
    electron abundance, and the helium mass fraction. It uses constants for the Boltzmann constant
    and proton mass.

    Args:
        U (numpy.ndarray): Particle internal energy in (km/s)^2.
        ne (numpy.ndarray): Electron abundance.

    Returns:
        numpy.ndarray: An array of temperatures for each particle in the snapshot.
    """
    BOLTZMANN = 1.38065e-16  # erg/K - NIST 2010
    PROTONMASS = 1.67262178e-24  # gram - NIST 2010

    yhelium = 0.0789  # Helium mass fraction
    T = U * (1.0 + 4.0 * yhelium) / (1.0 + yhelium + ne) * 1e10 * (2.0 / 3.0)
    T *= (PROTONMASS / BOLTZMANN)  # Convert to Kelvin
    return T

###############################################################################################

if __name__ == '__main__':
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python H5CosmoKit.py <path> <snapshot_numbers> <quantity>")
        sys.exit(1)

    # Extract arguments
    path = sys.argv[1]
    snapshot_numbers = [int(num) for num in sys.argv[2].split(',')]
    quantity = sys.argv[3]

    # Call the preview function
    preview(path, snapshot_numbers, quantity)
