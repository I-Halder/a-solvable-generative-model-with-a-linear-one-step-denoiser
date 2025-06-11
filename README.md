# **A solvable generative model with a linear, one-step denoiser**
This repository contains the code necessary to reproduce the results obtained by Indranil Halder in his recent paper - published at [ICML](https://openreview.net/forum?id=k4Q1ino3p0). 

## Experiments

The experiments on the linear diffusion model are run using `LDM.py`. The U-Net-based non-linear diffusion model with attention layers is defined in `model.py`, and `modules.py`. Simulations with it can be performed using `simulation.py`.

## Dependencies

We use Python 3.10.12, and dependencies with their exact
version numbers listed in `environment.yml`.

## Citing this work

```bibtex
@misc{halder2025solvablegenerativemodellinear,
      title={A solvable generative model with a linear, one-step denoiser}, 
      author={Indranil Halder},
      year={2025},
      eprint={2411.17807},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.17807}, 
}
```
