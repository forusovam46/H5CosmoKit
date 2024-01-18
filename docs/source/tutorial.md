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
import sys, os
sys.path.append(os.path.abspath("../../H5CosmoKit/"))
import H5CosmoKit
```

## Get the data

The output data is normally organised into 34 snapshots. You can download e.g. snap_010 (z = 2.00), snap_018 (z = 1.05) and snap_033 (z = 0.00) directly within this notebook.


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