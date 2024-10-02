# **Mathematical framework for memorization to generalization transition in diffusion-based generative models**
This repository contains the code necessary to reproduce the results obtained by Indranil Halder and Cengiz Pehlevan in [arxiv: ](https://arxiv.org). 

## Experiments

The experiments on the linear diffusion model are run using `LDM.py`. The U-Net-based non-linear diffusion model with attention layers is defined in `model.py`, and `modules.py`. Simulations with it can be performed using `simulation.py`.

## Dependencies

We use Python 3.10.12, and dependencies with their exact
version numbers listed in `environment.yml`.

## Citing this work

```bibtex
@Article{Halder2024,
  author  = {Halder, Indranil and Pehlevan, Cengiz},
  journal = {arxiv},
  title   = {On the onset of memorization to generalization transition in diffusion  models},
  year    = {2024},
  doi     = {}
}
```
