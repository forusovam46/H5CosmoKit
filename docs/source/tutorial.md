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
import H5CosmoKit as ckit
```

You can download e.g. snapshot_090 (z = 0.00) directly within this notebook.

```{code-cell}
urls = ["https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_0/snapshot_090.hdf5"] # extend the list as needed
local_files = ["snapshot_090.hdf5"]

for url, local_file in zip(urls, local_files):
    ckit.download_file(url, local_file)
```
## Density & Temperature

Now that we have the data, we can use the `H5CosmoKit` package to visualize density and temperature simple with `preview()`.

```{code-cell}
path = '.'  # Path to the snaps
snapshot_numbers = [90] # list of desired snapfile numbers

ckit.preview(path, snapshot_numbers, 'gas_density')
ckit.preview(path, snapshot_numbers, 'gas_temperature')
```

Or to `preview_3d` for interactive tree dimensional view.

```
subset_size = 300000
ckit.preview_3d(path, snapshot_numbers, 'gas_density', subset_size)
```
<iframe src="_static/Snapshot_90_at_z=0.00_gas_density.html" width="700" height="400"></iframe>
- [View the 3D Density Plot](_static/Snapshot_90_at_z=0.00_gas_density.html)

```
subset_size = 150000
ckit.preview_3d(path, snapshot_numbers, 'gas_temperature', subset_size)
```
- [View the 3D Temperature Plot](_static/Snapshot_90_at_z=0.00_gas_temperature.html)

## Soundspeed

```{code-cell}
path = '.'
snapshot_numbers = [90]
ckit.plot_soundspeed_distribution(path, snapshot_numbers, bw=0.6, x_limits=(0, 300), sample_size=10000)
```

## Power Spectra

As power spectra analysis uses Pylians, you might experience difficulties on machines other than Linux and Mac. For more details, visit the [Pylians documentation](https://pylians3.readthedocs.io/en/master/installation.html).

```{code-cell}
f_snap = './snapshot_090.hdf5'
snapshot_numbers = [90]
ckit.power_ratio(f_snap)

## Phase diagrams