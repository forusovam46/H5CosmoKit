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
#just testing example
import numpy as np
import matplotlib.pyplot as plt

# Generate 1000 random coordinates according to a normal (gaussian) distribution for x and for y
x = np.random.normal(0, 1, 1000)
y = np.random.normal(0, 1, 1000)
plt.plot(x,y, ".")
```

## Density
## Temperature
## Soundspeed
## Phase diagrams
## Power Spectra