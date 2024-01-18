---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial & Visualisation

Import modules:

```{code-cell}
import H5CosmoKit
import h5py
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import NearestNDInterpolator
```

## Get the data

The output data is normally organised into 34 snapshots. You can download e.g. snap_010 (z = 2.00), snap_018 (z = 1.05) and snap_033 (z = 0.00) directly within this notebook.

```{code-cell}
import requests

# URLs of the data files
urls =  ["https://users.flatironinstitute.org/\~camels/Sims/IllustrisTNG/CV/CV_0/snap_010.hdf5"] # extend the list as needed

# Local paths to save files
local_files = ["snap_010.hdf5"]

# Download files
for url, local_file in zip(urls, local_files):
    H5CosmoKit.download_file(url, local_file)

```
 If you use your own simulations output, use the following System of units (normally defined in param.txt) for correct output:
```
%---- System of units in CAMELS

UnitLength_in_cm          3.085678e21    %  kpc
UnitMass_in_g             1.989e43       %  1e10 solar masses
UnitVelocity_in_cm_per_s  1e5            %  1 km/sec
```
## Density & Temperature

Now that we have the data, we can use the `H5CosmoKit` package to visualize density and temperature.

```{code-cell}
# Example usage
path = '.'  # Path to the snaps
snapshot_numbers = [10]

H5CosmoKit.Preview(path, snapshot_numbers, 'rho_g')
H5CosmoKit.Preview(path, snapshot_numbers, 'temperature')
```

## Soundspeed
## Phase diagrams
## Power Spectra