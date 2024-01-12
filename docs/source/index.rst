.. H5CosmoKit documentation master file, created by
   sphinx-quickstart on Mon Jan  8 17:39:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to H5CosmoKit's documentation!
======================================

(come back later still in making)

`H5CosmoKit` is a toolkit designed for handling simulation data output in `HDF5` format, 
extracting and calculating quantities such as temperature, density, 
and power spectra from these files, and subsequently visualizing the results. 
It was put together to ease the entry into cosmological simulations for 
astrophysics students and cosmic enthusiasts.

This toolkit allows you to easily handle `HDF5` files such as `data<https://users.flatironinstitute.org/~camels/Sims/>`_ 
from projects like `CAMELS<https://www.camel-simulations.org/>`_ (Cosmology and Astrophysics with MachinE Learning Simulations) 
and therefore offer you a quick glance in simulation results. It can also be used to visualize your own simulation output from 
codes such as `Arepo<https://arepo-code.org/>`_. This project was assembled as part of my master's thesis, supervised by Univ.-Prof. Dr. Oliver Hahn 
(whose software `monofonIC<https://bitbucket.org/ohahn/monofonic/src/master/>`_ for creating initial conditions was also utilized.)
and my final project in a seminar `oss4astro<https://github.com/prashjet/oss4astro>`_ (Open Source Software Development for Astronomy) led by Prashin Jethwa, PhD.

It is being actively developed in a public 
`repository on GitHub
<https://github.com/forusovam46/H5CosmoKit.git>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   H5CosmoKit

Acknowledgments
======================================

This project incorporates code snippets or concepts inspired by the following projects:

- `CAMELS<https://github.com/franciscovillaescusa/CAMELS>`_: Licensed under the `MIT License<https://github.com/franciscovillaescusa/CAMELS/blob/master/LICENSE>`_.
- `Pylians3<https://github.com/franciscovillaescusa/Pylians3>`_: Licensed under the `MIT License<https://github.com/franciscovillaescusa/Pylians3/blob/master/LICENSE>`_.

I am grateful to the authors of these projects for their contributions to the open-source community.