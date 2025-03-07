# Field Matching: an Electrostatic Paradigm to Generate and Transfer Data

This is the official `Python` implementation of the paper  **Field Matching: an Electrostatic Paradigm to Generate and Transfer Data** (paper on [Arxiv](https://arxiv.org/pdf/2502.02367)) by [Alexander Kolesov](https://scholar.google.com/citations?user=vX2pmScAAAAJ&hl=ru&oi=ao), Stepan Manukhov , [Vladimir Palyulin](https://scholar.google.com/citations?user=IcjnBqkAAAAJ&hl=ru&oi=sra) and [Alexander Korotin](https://scholar.google.com/citations?user=1rIIvjAAAAAJ&hl=ru&oi=sra).

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

 