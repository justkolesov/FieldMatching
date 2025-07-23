# Field Matching: an Electrostatic Paradigm to Generate and Transfer Data (ICML 2025)

This is the official `Python` implementation of the [ICML 2025](https://icml.cc/virtual/2025/poster/46213) paper  **Field Matching: an Electrostatic Paradigm to Generate and Transfer Data** by [Alexander Kolesov](https://scholar.google.com/citations?user=vX2pmScAAAAJ&hl=ru&oi=ao), Stepan Manukhov , [Vladimir V. Palyulin](https://scholar.google.com/citations?user=IcjnBqkAAAAJ&hl=ru&oi=sra) and [Alexander Korotin](https://scholar.google.com/citations?user=1rIIvjAAAAAJ&hl=ru&oi=sra).

The repository contains reproducible PyTorch source code for computing maps for noise-to-data as well as data-to-data scenarios in high dimensions with neural networks. Examples are provided for toy 3D problems, unconditional data generation and unpaired translation problems.

## Pre-requisites

The implementation is GPU-based. Single GPU GTX 1080 ti is enough to run each particular experiment. We tested the code with `torch==2.1.1+cu121`. The code might not run as intended in older/newer `torch` versions. Versions of other libraries are specified in `requirements.txt`. 

 
## Repository structure

All the experiments are issued in the form of pretty self-explanatory jupyter notebooks.

- `models.py` - auxiliary source code for the constructing of the neural network model.
- `ToyExperiments.ipynb` - the notebook of 2D illustrative example
- `Generating.ipynb` - the notebook of the noise-to-image generation.
- `Translation.ipynb` -  the notebook of Image-to-Image translation.
 

```console
pip install -r requirements.txt
```
- Download  [MNIST](https://yann.lecun.com/exdb/mnist) dataset

- Set downloaded dataset in appropriate subfolder in `data/`.

 
