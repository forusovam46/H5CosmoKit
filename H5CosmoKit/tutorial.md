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

Test `matplotlib` 

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.interpolate import NearestNDInterpolator
from matplotlib.colors import LogNorm
```

## Get the data

The output data is normally organised into 34 snapshots. Download e.g. snap_010 (z = 2.00), snap_018 (z = 1.05) and snap_033 (z = 0.00) from [Flatiron Institute](https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_0/). If you use your own simulations output, use the following System of units (normally defined in param.txt):

```
%---- System of units in CAMELS

UnitLength_in_cm          3.085678e21    %  kpc
UnitMass_in_g             1.989e43       %  1e10 solar masses
UnitVelocity_in_cm_per_s  1e5            %  1 km/sec
```
## Density & Temperature

```{code-cell}
import h5py
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import NearestNDInterpolator
import H5CosmoKit

# Example usage
'../camelsdata'
snapshot_numbers = [10]

H5CosmoKit.Preview(path, snapshot_numbers, 'rho_g')
H5CosmoKit.Preview(path, snapshot_numbers, 'temperature')
```

## Soundspeed
## Phase diagrams
## Power Spectra