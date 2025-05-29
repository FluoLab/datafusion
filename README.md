# datafusion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15496000.svg)](https://doi.org/10.5281/zenodo.15496000)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Fusing multidimensional data together, with a focus on microscopy.
This repo allows you to fuse time-resolved multispectral single-pixel camera data (4D) with
structured illumination microscopy (SIM) (3D) data acquired with CMOS or CCD cameras to reconstruct
high-resolution fluorescence lifetime multispectral data (5D).

https://github.com/user-attachments/assets/f0ec59b7-89d7-4ceb-8ca2-f67bbd98fbae

## Table of Contents

- ⚙️ [Instructions](#instructions)
- ℹ️ [How does it work?](#how-does-it-work)
- ⚡️ [How to use it?](#how-to-use-it)

## Instructions
Use conda to create an environment with the required dependencies.

```bash
# Clone it
git clone https://github.com/FluoLab/datafusion

# Enter
cd datafusion

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate datafusion
```

## How does it work?

Data fusion algorithms in this repo solve the following convex optimization problem:  
$$
\underset{\mathbf{X}}{\arg\min} \quad \mathcal{L}(\mathbf{X}) =
\frac{w_1}{2}\|ST\mathbf{X} - \mathbf{Y}_{cmos}\|_2^2 +
\frac{w_2}{2}\|RD\mathbf{X} - \mathbf{Y}_{spc}\|_2^2,
$$  
where:

* $\mathbf{X}$ is the 5D fluorescence time-resolved multispectral volume we are reconstructing
* $\mathbf{Y}_{cmos}$ is the structured illumination microscopy 3D data
* $\mathbf{Y}_{spc}$ is the single-pixel camera 4D data
* $T$, $S$ and $D$ are the summation operators over different dimensions,
  temporal decay, spectrum, and depth dimensions, respectively
* $R$ is the downsampling operator in 2D.

`fusion.py` provides two ways to solve the optimization problem:

* Linear Conjugate Gradient: best for convergence speed of this problem
* Adam + Backpropagation: great for inclusion of learned reconstruction techniques

## How to use it?

```python
from datafusion.fusion import FusionCG
from datafusion.utils import (
    download_url,
    load_data,
    ZENODO_URL,
    RESOURCES_PATH,
)

# Download the example data.
download_url(
    ZENODO_URL,
    RESOURCES_PATH / "acquisitions.zip",
    chunk_size=2048,
    unzip=True,
)

# Load the data.
cmos, spc, _, _ = load_data(
    path=RESOURCES_PATH / "acquisitions" / "cells" / "cells_0.50cr.npz",
    max_xy_size=128,
)

# Set up the penalties for the fidelity terms.
weights = {
    "spatial": 0.5,
    "spectro_temporal": 0.5,
}

# Fuse the data.
fuse_with_cg = FusionCG(
    spc,
    cmos,
    weights=weights,
    init_type="baseline",
    device="cpu",  # "cuda" or "mps" for GPU acceleration.
)
x, _, _ = fuse_with_cg(max_iterations=20)
# x is now the reconstructed 5D fluorescence time-resolved multispectral volume.

```

## Acknowledgements

This work is supported by the European Union (GA 101072354) 🇪🇺
and the UK Research and Innovation (EP/X030733/1) 🇬🇧. It comes from
the effort and synergy of multiple people from different fields:
Federico Simon, Serban Cristian Tudosie, Shivaprasad Varakkoth, 
Valerio Gandolfi, Simon Arridge,
Andrea Farina, and Cosimo D'Andrea.