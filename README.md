<p align="center">
  <img width="340" src="https://github.com/forusovam46/H5CosmoKit/blob/main/docs/source/_static/logo.png">
</p>

![GitHub Actions Status](https://github.com/forusovam46/H5CosmoKit/actions/workflows/main.yml/badge.svg)
[![codecov](https://codecov.io/gh/forusovam46/H5CosmoKit/branch/main/graph/badge.svg)](https://codecov.io/gh/forusovam46/H5CosmoKit)
[![Documentation Status](https://readthedocs.org/projects/h5cosmokit/badge/?version=latest)](https://h5cosmokit.readthedocs.io/en/latest/?badge=latest)

(come back later still in making)

`H5CosmoKit` is a toolkit designed for handling simulation data output in `HDF5` format, 
extracting and calculating quantities such as temperature, density, 
and power spectra from these files, and subsequently visualizing the results. 
It was put together to ease the entry into cosmological simulations for 
astrophysics students and cosmic enthusiasts.

This toolkit allows you to easily handle `HDF5` files such as [data](https://users.flatironinstitute.org/~camels/Sims/) from projects like [CAMELS](https://www.camel-simulations.org/) (Cosmology and Astrophysics with MachinE Learning Simulations) and therefore offer you a quick glance in simulation results. It can also be used to visualize your own simulation output from codes such as [Arepo](https://arepo-code.org/). This project was assembled as part of my master's thesis, supervised by Univ.-Prof. Dr. Oliver Hahn 
(whose software [monofonIC](https://bitbucket.org/ohahn/monofonic/src/master/) for creating initial conditions was also utilized.)
and my final project in a seminar [oss4astro](https://github.com/prashjet/oss4astro) (Open Source Software Development for Astronomy) led by Prashin Jethwa, PhD.

Read the full documentation at [h5cosmokit.readthedocs.io](https://h5cosmokit.readthedocs.io/).

## Installation

Install with pip
`pip install H5CosmoKit`
or clone the repository and install in editable mode
`cd H5CosmoKit` and
`pip install -e .`

## Acknowledgments

This project incorporates code snippets or concepts inspired by the following projects:

- [CAMELS](https://github.com/franciscovillaescusa/CAMELS): Licensed under the [MIT License](https://github.com/franciscovillaescusa/CAMELS/blob/master/LICENSE).
- [Pylians3](https://github.com/franciscovillaescusa/Pylians3): Licensed under the [MIT License](https://github.com/franciscovillaescusa/Pylians3/blob/master/LICENSE).

I am grateful to the authors of these projects for their contributions to the open-source community.
