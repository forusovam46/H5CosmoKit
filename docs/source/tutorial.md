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

## Get the data

Import the module:

```{code-cell}
import H5CosmoKit
```

You can download e.g. snap_010 (z = 2.00), snap_018 (z = 1.05) and snap_033 (z = 0.00) directly within this notebook.

```{code-cell}
urls = ["https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_0/snap_033.hdf5"] # extend the list as needed
local_files = ["snap_033.hdf5"]

for url, local_file in zip(urls, local_files):
    H5CosmoKit.download_file(url, local_file)
```
If you use your own simulations output, use the following System of units (normally defined in param.txt) for correct units in plots:
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
path = '.'  # Path to the snaps e.q. in windows r'C:\Users\<username>\...\snapfiles'
snapshot_numbers = [33]

H5CosmoKit.preview(path, snapshot_numbers, 'gas_density')
H5CosmoKit.preview(path, snapshot_numbers, 'gas_temperature')
```

Or preview the data in 3D.

```{code-cell}
subset_size = 300000
H5CosmoKit.preview_3d(path, snapshot_numbers, 'gas_density', subset_size)
```

```{code-cell}
subset_size = 150000
H5CosmoKit.preview_3d(path, snapshot_numbers, 'gas_temperature', subset_size)
```

## Soundspeed
## Phase diagrams
## Power Spectra