# A Solvable Generative Model with a Linear, One-Step Denoiser

This repository provides the accompanying code for [A solvable generative model with a linear, one-step denoiser](https://openreview.net/forum?id=k4Q1ino3p0), presented at High-dimensional Learning Dynamics @ ICML 2025. The core experiments are implemented in PyTorch with PyTorch Lightning for training and logging utilities.

---

## Installation

The project uses a single conda environment defined in `environment.yml` (Python 3.10, PyTorch, PyTorch Lightning, and supporting libraries).

### Create and activate the environment

```bash
# From the repository root
conda env create -f environment.yml
conda activate environment
```

The `prefix` field at the bottom of `environment.yml` is a placeholder for a local path; edit it to match your preferred environment location.

---

## Repository Structure

```text
a-solvable-generative-model-with-a-linear-one-step-denoiser/
├── LDM.py
├── model.py
├── modules.py
├── simulation.py
├── environment.yml
├── LICENSE
└── README.md
```

- `LDM.py`:  
  Standalone script for the analytically tractable linear diffusion model.  
  Generates synthetic data, applies the linear denoiser, and computes distributional distances (e.g., Hellinger-type metrics) as a function of the number of training samples.

- `simulation.py`:  
  Main simulation script for the learned diffusion model:
  - Defines the Gaussian (or Gaussian mixture) data-generating process.
  - Trains a diffusion model using PyTorch Lightning.
  - Generates samples and reports distances between original, training, and generated data across sample sizes.

- `model.py`:  
  Defines the `DiffusionModel` Lightning module:
  - Sets up the diffusion schedule and noise process.
  - Wraps a U-Net–style backbone (from `modules.py`).
  - Implements the training and sampling logic.

- `modules.py`:  
  Neural network building blocks used by `DiffusionModel`, including:
  - Self-attention layers.
  - U-Net components (`DoubleConv`, `Down`, `Up`, `OutConv`, etc.).

- `environment.yml`:  
  Exact conda environment (Python version and package pins).

- `LICENSE`:  
  License information for using and modifying this code.

---

## Simulation Scripts

The main scripts are `LDM.py` and `simulation.py`. Both can be run directly and rely on hyperparameters set at the top of each file.

### Linear diffusion model (`LDM.py`)

This script reproduces the core experiments for the analytically tractable linear diffusion model.

Key configuration (at the top of `LDM.py`):

- `mean_list`, `std_list`: mean and standard deviation of the underlying Gaussian distribution(s).
- `d`: number of pixels per sample `d = n * n`.
- `num_total`: total number of data points.
- `num_samples_start`, `num_samples_end`, `num_samples_difference`: range and step size of training sample counts.
- Other internal parameters controlling the forward and reverse dynamics and distance computations.

Usage:

```bash
python LDM.py
```

Behavior:

- Draws samples from the specified Gaussian distribution(s).
- Applies the forward and reverse linear diffusion dynamics.
- Computes distributional distances between:
  - The original distribution.
  - The distribution of training data.
  - The distribution of generated samples.
- Prints, for each training sample size, statistics such as:
  - Average distance between original and generated distributions.
  - Average distance between training and generated distributions.
  - Average Hellinger distance squared.
  - Probability that a “Delta” statistic is positive.

All results are printed to standard output.

### Learned diffusion model simulations (`simulation.py`)

This script trains a diffusion model (U-Net–style backbone with self-attention) on synthetic Gaussian or Gaussian mixture data and evaluates how generated samples compare to original and training data.

Key configuration (at the top of `simulation.py`):

- `max_epoch`: number of training epochs.
- `diffusion_steps`: number of diffusion time steps.
- `batch_size`: training batch size.
- `num_generated`: number of generated samples per configuration.
- `mean_list`, `std_list`, `weights`: parameters of the Gaussian mixture.
- `n`: number of pixels per side (multiple of 8).
- `chunk_cutoff`: maximum number of data points used when computing distances.
- `num_total`: total number of data points to sample.
- `num_samples_start`, `num_samples_end`, `num_samples_difference`: training sample range and step.
- `num_simulation`: number of repeated simulations for averaging statistics.

The script:

1. Constructs a Gaussian (mixture) dataset in `n × n` images.
2. Splits data into “train” and “original” sets depending on a fraction `frac`.
3. Wraps data in the `diffSet` dataset class and PyTorch `DataLoader`s.
4. Instantiates `DiffusionModel` from `model.py`.
5. Trains using `pl.Trainer` (PyTorch Lightning).
6. Generates samples and computes:
   - Distances between original and generated data.
   - Distances between train and generated data.
   - A “Delta” statistic and its sign.
7. Repeats across multiple training sample sizes and simulations, printing:
   - Average value of Delta.
   - Probability that Delta > 0.
   - Average distances for each sample size.

Basic usage:

```bash
python simulation.py
```

To modify the experiment (e.g., more mixture components, different sample sizes, longer training), edit the corresponding variables at the top of `simulation.py` and rerun.

---

## Model and Architecture (`model.py` and `modules.py`)

`model.py` defines the `DiffusionModel` Lightning module:

- Components:
  - Diffusion schedule (`beta_small`, `beta_large`).
  - U-Net–style encoder–decoder with skip connections.
  - Self-attention blocks for capturing global structure.
- Methods:
  - Forward pass used for predicting the noise / denoised sample.
  - Training loop hooks for PyTorch Lightning.

`modules.py` provides the building blocks:

- `SelfAttention`, `SAWrapper`: multi-head self-attention layers.
- U-Net components:
  - `DoubleConv`
  - `Down`
  - `Up`
  - `OutConv`

You can modify these modules to explore alternative architectures while keeping the surrounding simulation code unchanged.

---

## Additional Commands

If you use TensorBoard to inspect logs:

```bash
tensorboard --logdir lightning_logs/
```

---

## Citation

```bibtex
@inproceedings{
halder2025a,
title={A solvable generative model with a linear, one-step denoiser},
author={Indranil Halder},
booktitle={High-dimensional Learning Dynamics 2025},
year={2025},
url={https://openreview.net/forum?id=k4Q1ino3p0}
}

```

