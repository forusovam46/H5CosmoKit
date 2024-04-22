import h5py
from matplotlib.colors import LogNorm
import MAS_library as MASL
import matplotlib.pyplot as plt
import numpy as np
import os
import Pk_library as PKL
import plotly.graph_objects as go
import ptitprince as pt
import requests
from scipy.interpolate import NearestNDInterpolator
import seaborn as sns
import sys

###############################################################################################

def calc_soundSpeed(U):
    """
    Calculates the sound speed for an ideal gas given the specific internal energy.

    The formula used assumes an adiabatic index (gamma) of 5/3. 
    The sound speed is calculated using the simplified formula derived under the assumption of a 
    constant volume (isochoric) process, where the density dependency can be ignored.

    Args:
        U (numpy.ndarray): An array of specific internal energies for which the sound speeds are calculated.

    Returns:
        numpy.ndarray: An array of sound speeds corresponding to the input specific internal energies.

    Note:
        This function is derived from the more general form:
        Cs = sqrt(gamma * P / rho)
        Where P = (gamma - 1) * rho * U is the pressure of the gas.
        In contexts where density (rho) is constant or its effect is normalized out,
        the formula simplifies to only depend on the specific internal energy (U):
        Cs = sqrt(gamma * (gamma - 1) * U)
    """
    gamma = 5/3  # Adiabatic index for monatomic ideal gas
    Cs = np.sqrt(gamma * (gamma - 1.0) * U)  # Calculate sound speed
    return Cs

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

###############################################################################################
#TODO: revise functionality for power spectra
def power_ratio(f_snap):
    """
    Processes a snapshot file to compute and plot power spectrum ratios.
    """
    data = h5py.File(f_snap, 'r')
    BoxSize = data['Header'].attrs['BoxSize'] / 1e3  # Mpc/h
    Masses = data['Header'].attrs['MassTable'] * 1e10  # Msun/h

    # Read baryon (gas) and CDM particles
    pos_baryons, mass_baryons = read_particles(data, 0, Masses)  # Assuming gas for baryons
    pos_dm, mass_dm = read_particles(data, 1, Masses)  # CDM

    # Compute power spectra
    k_baryon, Pk_baryon = compute_power_spectrum(pos_baryons, mass_baryons, BoxSize)
    k_dm, Pk_dm = compute_power_spectrum(pos_dm, mass_dm, BoxSize)

    # Plot the ratio
    plot_power_spectrum_ratio(k_baryon, Pk_baryon, k_dm, Pk_dm)

def read_particles(data, part_type, mass_table):
    """
    Reads positions and masses of particles of a given type from the HDF5 data.
    """
    pos = data[f'PartType{part_type}/Coordinates'][:] / 1e3  # Positions in Mpc/h
    try:
        mass = data[f'PartType{part_type}/Masses'][:] * 1e10  # Masses in Msun/h
    except KeyError:
        mass = np.ones(len(pos)) * mass_table[part_type]  # Uniform mass
    return pos.astype(np.float32), mass.astype(np.float32)

def compute_power_spectrum(pos, mass, BoxSize, grid=512, MAS='CIC'):
    """
    Computes the power spectrum for given positions and masses of particles.
    """
    delta = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(pos, delta, BoxSize, MAS, W=mass, verbose=True)

    # Normalize the density field
    delta /= np.mean(delta, dtype=np.float32)
    delta -= 1.0

    # Compute the Power Spectrum
    axis = 0
    threads = 1
    verbose = True
    Pk_class = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)
    k = Pk_class.k3D
    Pk = Pk_class.Pk[:, 0]

    return k, Pk

def plot_power_spectrum_ratio(k_baryon, Pk_baryon, k_dm, Pk_dm):
    """
    Plots the ratio of power spectra.
    """
    plt.figure(figsize=(8, 6))
    plt.loglog(k_baryon, Pk_baryon / Pk_dm, 'o', label='P_baryon / P_CDM', alpha=0.8)  # Plot as points
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.8)  # Dashed line at y=1
    plt.xlabel(r'$k \, [h/\mathrm{Mpc}]$')
    plt.ylabel(r'$P_{\mathrm{baryon}}(k) / P_{\mathrm{CDM}}(k)$')
    plt.legend()
    plt.show()

###############################################################################################

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
        quantity (str): Quantity name (e.g., 'gas_density' or 'gas_temperature').

    Returns:
        matplotlib.image.AxesImage: The plotted image.

    # Example usage
        path = '/gpfs/data/fs72085/mfo/CAMELS/CV0_CAMELS_output'
        snapshot_numbers = [10, 18, 33]
        plot_sound_speeds(path, snapshot_numbers, sample_size=10000)  # Specifying a sample size
    """

    cmap = plt.cm.get_cmap('Spectral').reversed() if quantity == 'gas_density' else plt.cm.get_cmap('hot')

    im = ax.imshow(grid_quantity[:, :, 128], extent=[0, boxSize, 0, boxSize], origin='lower', norm=LogNorm(), cmap=cmap)
    ax.axis('equal')
    ax.set_xlabel(r'$x \:[Mpc/h$]')
    ax.set_ylabel(r'$y \:[Mpc/h$]')
    ax.set_title(title)
    return im

def plot_soundspeed_distribution(path, snapshot_numbers, bw=0.7, x_limits=None, sample_size=None):
    """
    Plots the distribution of sound speeds from multiple snapshot files.

    Args:
        path (str): The base path containing the snapshot files.
        snapshot_numbers (list of int): List of snapshot numbers to plot.
        bw (float): Bandwidth for the density estimation in the raincloud plot. Default is 0.6.
        x_limits (tuple): X-axis limits for the plot. Default is None, which auto-scales.
        sample_size (int): Number of random samples to select from each snapshot. Default is None (use all data).

    Example:
        path = '/gpfs/data/fs72085/mfo/CAMELS/CV0_CAMELS_output/'
        snapshot_numbers = [10, 18, 33]
        plot_soundspeed_distribution(path, snapshot_numbers, bw=0.9, x_limits=(0, 300), sample_size=10000)
    """
    def calc_soundSpeed(U):
        gamma = 5/3  # Adiabatic index for monatomic ideal gas
        Cs = np.sqrt(gamma * (gamma - 1.0) * U)  # Calculate sound speed
        return Cs

    Cs_means = []
    all_soundspeeds = []
    soundspeed_lengths = []
    redshifts = []
    scale_factors = []

    # Process each snapshot
    for num in snapshot_numbers:
        snapshot_path = os.path.join(path, f'snap_{num:03}.hdf5')
        boxSize, redshift, scale_factor, pos_g, rho_g, U, ne = read_snapshot(snapshot_path)
        
        if sample_size is not None:
            random_indices = np.random.choice(len(U), min(sample_size, len(U)), replace=False)
            U = U[random_indices]

        soundspeed = calc_soundSpeed(U)
        Cs_mean = np.mean(soundspeed)
        Cs_means.append(Cs_mean)
        all_soundspeeds.extend(soundspeed)
        soundspeed_lengths.append(len(soundspeed))
        redshifts.append(round(redshift))
        scale_factors.append(scale_factor)

    # Convert collected soundspeeds and corresponding redshifts for plotting
    soundspeed_data = np.repeat(redshifts, soundspeed_lengths)

    # Create the raincloud plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    kwargs = {"rain_alpha": 0.3}
    pt.RainCloud(x=soundspeed_data, y=all_soundspeeds, palette="Set2", bw=bw, width_viol=0.8,
                 ax=ax, orient='h', pointplot=True, **kwargs)
    if x_limits is not None:
        plt.xlim(x_limits)
    plt.title("Raincloud plot of Sound Speed across Redshifts")
    plt.ylabel("Redshift")
    plt.xlabel("Sound Speed [km/s]")

    # Annotate mean sound speeds
    for mean, redshift in zip(Cs_means, redshifts):
        ax.text(mean + 7, redshift + 0.2, f'$\overline{{c}}_s = {mean:.2f}$ km/s', color='black')

    # Fit and plot polynomial line
    coef = np.polyfit(scale_factors, Cs_means, 1)
    poly1d_fn = np.poly1d(coef)
    polynomial_str = f"Best linear fit: $\overline{{c}}_s = {coef[0]:.2f}a + {coef[1]:.2f}$"
    plt.text(0.65, 0.05, polynomial_str, transform=ax.transAxes)

    plt.show()

def preview(path, snapshot_numbers, quantity, unit_scale='kpc'):
    """
    Creates a plot of snapshot quantity from multiple snapshot files.

    Args:
        path (str): The base path containing the snapshot files.
        snapshot_numbers (list of int): List of snapshot numbers to plot.
        quantity (str): Quantity name (e.g., 'gas_density' or 'gas_temperature').
        unit_scale (str): Unit scale ('kpc' or 'mpc') for interpreting the data. Default is 'kpc'.

    Example:
        from H5CosmoKit import preview

        # Example usage to visualize gas density from a snapshot in a cosmological simulation
        path = '/your/data/directory'
        snapshot_numbers = [10, 20, 30]  # Example with multiple snapshots
        quantity = 'gas_density'

        preview(path, snapshot_numbers, quantity, unit_scale='Mpc')
    """
    if not snapshot_numbers:
        print("No snapshot numbers provided.")
        return

    quantity = validate_quantity(quantity)

    fig, axes = plt.subplots(1, len(snapshot_numbers), figsize=(7 * len(snapshot_numbers), 5))
    if len(snapshot_numbers) == 1:
        axes = [axes]  # Make sure axes is always a list

    path_last_parts = os.path.normpath(path).split(os.path.sep)[-2:]
    subtitle = "_".join(path_last_parts)
    fig.suptitle(subtitle)

    first_time = True
    for ax, snapshot_number in zip(axes, snapshot_numbers):
        snapshot_path = os.path.join(path, f'snap_{snapshot_number:03}.hdf5')

        # Read data from snapshot
        boxSize, redshift, scale_factor, pos_g, rho_g, U, ne = read_snapshot(snapshot_path, unit_scale)

        if quantity == 'gas_temperature':
            T = temperature(U, ne)  # Calculate temperature
            quantity_g = T
            colorbar_label = "T [K]"
        else:  # Assuming 'gas_density'
            quantity_g = rho_g
            colorbar_label = r'Density [$\mathrm{10^{10} \, M_\odot/h/(cMpc/h)^3}$]'

        interp_quantity = interpolate_quantity(pos_g, quantity_g, boxSize)
        xx = np.linspace(0, boxSize, 256, endpoint=False)
        grid_x, grid_y, grid_z = np.meshgrid(xx, xx, xx)
        grid_quantity = interp_quantity((grid_x, grid_y, grid_z))
        title = f'z={round(redshift)}'
        im = plot_snapshot(ax, grid_quantity, boxSize, title, quantity)  # Pass the correct ax

        if first_time:
            f = im
            first_time = False

    cbar = fig.colorbar(f, ax=axes, label=colorbar_label)  # Add colorbar to the figure, not to each subplot

    save_plot(fig, path, quantity)

def preview_3d(path, snapshot_numbers, quantity, subset_size, unit_scale='kpc'):
    """
    Creates an interactive 3D plot of snapshot quantity using Plotly and saves it as an HTML file.

    Args:
        path (str): The base path containing the snapshot files.
        snapshot_numbers (list of int): List of snapshot numbers to plot.
        quantity (str): Quantity name (e.g., 'gas_density' or 'gas_temperature').
        subset_size (int): Size of randomly chosen points to be plotted to manage performance.

    Example Usage:
        path = '/your/data/directory'
        snapshot_numbers = [30]
        quantity = 'gas_temperature'
        subset_size = 10000

        preview_3d(path, snapshot_numbers, quantity, subset_size, unit_scale='Mpc')

    """
    # Validate input
    if not snapshot_numbers:
        print("No snapshot numbers provided.")
        return
    
    quantity = validate_quantity(quantity)

    # Prepare the object
    for snapshot_number in snapshot_numbers:
        snapshot_path = os.path.join(path, f'snap_{snapshot_number:03}.hdf5')

        # Read data from snapshot
        boxSize, redshift, scale_factor, pos_g, rho_g, U, ne = read_snapshot(snapshot_path, unit_scale)

        # Select a subset of points for plotting to manage performance
        plot_subset = np.random.choice(len(pos_g), size=min(len(pos_g), subset_size), replace=False)

        if quantity == 'gas_temperature':
            title = f'Snapshot {snapshot_number} at z={redshift:.2f}'
            quantity_g = temperature(U, ne)  # Calculate temperature
            colorbar_title = "Temperature [K]"
            colormap = 'Hot'
        else:  # Assuming 'gas_density'
            title = f'Snapshot {snapshot_number} at z={redshift:.2f}'
            quantity_g = rho_g
            colorbar_title = 'Log(ρ<sub>g</sub>) [$\mathrm{10^{10}M_☉/h/(cMpc/h)<sup>3</sup>]'
            colormap = 'Spectral_r'

        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=pos_g[plot_subset, 0],
            y=pos_g[plot_subset, 1],
            z=pos_g[plot_subset, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=np.log10(quantity_g[plot_subset]),  # Set color to log scale of quantity
                colorscale=colormap,
                colorbar=dict(title=colorbar_title),
                opacity=0.1
            ),
            name=title
        )])

        # Update plot layout
        fig.update_layout(
            title="Snapshot Data",
            scene=dict(
                xaxis_title='X [Mpc/h]',
                yaxis_title='Y [Mpc/h]',
                zaxis_title='Z [Mpc/h]',
            ),
            margin=dict(r=0, b=0, l=0, t=0),
            template="plotly_dark"
        )

        # Show plot
        fig.show()

        # Save Plotly figure as HTML file
        filename = f"{title}_{quantity}.html".replace(" ", "_")
        fig.write_html(filename)

def read_snapshot(snapshot_path, unit_scale='kpc'):
    """
    Reads snapshot data from a HDF5 file, converting units if required.
    
    Args:
        snapshot_path (str): Path to the HDF5 snapshot file.
        unit_scale (str): Unit scale for length ('kpc' or 'mpc') of your snapshot file, default is 'mpc'.
    
    Returns:
        tuple: Contains the box size, redshift, scale_factor, positions, densities, internal energies,
               and electron abundances (if available) adjusted to the desired unit scale.
               
    System of units for default option (mpc):
    %---- System of units
    UnitLength_in_cm                      3.085678e24    %  1.0 Mpc
    UnitMass_in_g                         1.989e43       %  1.0e10 solar masses
    UnitVelocity_in_cm_per_s              1e5            %  1 km/sec
    """
    factor = 1e-3 if unit_scale == 'kpc' else 1  # Convert from kpc to Mpc if needed i.e for CAMELS
    density_factor = factor**3  if unit_scale == 'kpc' else 1 # Adjust density for the change in volume unit

    with h5py.File(snapshot_path, 'r') as f:
        boxSize = f['Header'].attrs['BoxSize'] * factor
        redshift = f['Header'].attrs['Redshift']
        scale_factor = f['Header'].attrs[u'Time']
        pos_g = f['PartType0/Coordinates'][:] * factor
        pos_g = pos_g.astype(np.float32)
        rho_g = f['PartType0/Density'][:] / density_factor
        U = f['PartType0/InternalEnergy'][:]
        ne = f['PartType0/ElectronAbundance'][:] if 'PartType0/ElectronAbundance' in f else np.zeros_like(U)

    return boxSize, redshift, scale_factor, pos_g, rho_g, U, ne

def save_plot(fig, path, quantity):
    """
    Saves the figure to a file with an appropriate filename.

    Args:
        fig (matplotlib.figure.Figure): The figure to be saved.
        path (str): The base path where the plot will be saved.
        quantity (str): Quantity name (e.g., 'gas_density' or 'gas_temperature').
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

def validate_quantity(quantity):
    """
    Validates the provided quantity against allowed values.

    Args:
        quantity (str): The quantity to validate.

    Returns:
        str: The validated quantity.
    """
    valid_quantities = ['gas_temperature', 'gas_density']
    if quantity not in valid_quantities:
        print("You have entered an invalid quantity. Please choose from 'gas_temperature' or 'gas_density'.")
        quantity = input("Enter your choice (gas_temperature/gas_density): ").strip()
        if quantity not in valid_quantities:
            print("Invalid input received. Defaulting to 'gas_density'.")
            quantity = 'gas_density'
    return quantity

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
