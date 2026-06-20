# Autoencoder From Scratch

This project is a notebook-based implementation of a simple fully connected autoencoder for the MNIST dataset. It trains a PyTorch model to compress handwritten digits into a low-dimensional latent representation, reconstruct the input images, and visualize the learned embedding space.

## What It Does

- Loads the MNIST dataset with `torchvision`
- Trains a feed-forward autoencoder in PyTorch
- Reconstructs input digits and plots original vs. reconstructed images
- Extracts latent vectors from the encoder
- Visualizes the latent space with Matplotlib or Plotly
- Computes a silhouette score to estimate class separation in latent space
- Generates digits from random latent vectors using the decoder
- Trains a denoising autoencoder (DAE) on Gaussian-noise-corrupted digits and visualizes the cleaned-up reconstructions

## Project Files

- [autoencoder_from_scratch.ipynb](autoencoder_from_scratch.ipynb): main notebook containing the model, training loop, and visualizations
- [pyproject.toml](pyproject.toml): project dependencies, managed with [uv](https://docs.astral.sh/uv/)

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management
- An NVIDIA GPU with CUDA 12.4-compatible drivers (optional, falls back to CPU)

## Setup

```bash
uv sync
```

This creates a `.venv` and installs all dependencies, including a CUDA-enabled build of PyTorch.

## How To Run

1. Open [autoencoder_from_scratch.ipynb](autoencoder_from_scratch.ipynb) in Jupyter or VS Code, selecting the `.venv` kernel.
2. Run the notebook cells from top to bottom.
3. Let MNIST download into the local `./data` folder the first time you run it.
4. Review the printed training loss, reconstruction plots, latent-space plots, and the silhouette score.

## Model Overview

The notebook defines a compact autoencoder with:

- An encoder that maps flattened 28x28 images into a latent vector
- A decoder that reconstructs the image from that latent vector
- ReLU activations in the hidden layers
- A sigmoid output layer to keep reconstructed pixel values in the $[0, 1]$ range

The model is trained with mean squared error loss and the Adam optimizer.

## Outputs

When you run the notebook, you should see:

- Training loss printed for each epoch
- A side-by-side comparison of original and reconstructed MNIST digits
- A 1D, 2D, or 3D latent-space visualization, depending on the latent dimension
- A silhouette score for the learned embeddings
- A sample reconstruction generated from a random latent vector

All generated plots are also saved in the `results/` folder:

- `reconstructions.png`: original vs. reconstructed digits
- `latent_space_*.png`: latent-space visualization (filename depends on latent dimensionality)
- `latent_space_pca_3d.html`: interactive, rotatable 3D PCA projection of the latent space (Plotly)
- `random_generation.png`: reconstruction from a random latent vector
- `denoising_reconstructions.png`: noisy inputs vs. denoising-autoencoder reconstructions

## Notes

- The notebook is intended as an educational example rather than a production training pipeline.
