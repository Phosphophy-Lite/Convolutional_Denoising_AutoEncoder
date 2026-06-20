# Convolutional Denoising Autoencoder for medical chest X-ray images (ChestMNIST)

## Overview

This Jupyter notebook builds and trains **a convolutional denoising autoencoder (CDAE)** on the
ChestMNIST dataset (from the MedMNIST collection). It covers the full
pipeline:

1. Loading the ChestMNIST dataset and injecting synthetic noise
   (gaussian, poisson, salt & pepper) into the images.
2. Two model architectures:
   - A configurable plain Convolutional Autoencoder (CDAE).
   - A "U-Net" style autoencoder : mirror of the previous architecture but with skip connections. (*! Not the canonical U-Net architecture as there are no double convolutional layers !*)
3. Training either model in "specialist" mode (one fixed noise type/level on the training images)
   or "generalist" mode (trained across multiple noise types/levels).
4. Evaluating the trained model against classical denoising filters
   (e.g. median denoising filters or NL-Means) using PSNR, SSIM, MSE, MS-SSIM metrics.
5. Visualizing denoising results and robustness of the trained model across noise levels.

The notebook is self-contained: run the cells from top to bottom, choosing
the options described in each section's markdown cell as you go.

The notebook can be important in a Google Colab.

## Main library requirements

Assuming Python 3.11+.

The full dependency list is in [requirements.txt](requirements.txt), but the key
libraries this notebook directly depends on are:

- `torch` (PyTorch - model definition, training, GPU/MPS support)
- `torchvision` (image transforms)
- `medmnist` (ChestMNIST dataset loader)
- `numpy` (array/tensor manipulation)
- `matplotlib` (plotting losses, images, metrics)
- `scikit-image` (image processing utilities)
- `scikit-learn` (evaluation helpers)
- `scipy` (classical filters used as denoising baselines)
- `pytorch-msssim` (MS-SSIM loss function option)

## Installing requirements

It is recommended to use a virtual environment before installing (unless using Google Colab).

### 1) Create and activate a virtual environment

**Windows (PowerShell or cmd):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install the dependencies from requirements.txt

**Windows / macOS / Linux** (same command once the venv is active):

```bash
pip install -r requirements.txt
```


> **Notes**
> - GPU acceleration: the notebook auto-detects CUDA (NVIDIA GPUs) and MPS
>   (Apple Silicon) via `torch.cuda.is_available()` / `torch.mps.is_available()`,
>   falling back to CPU otherwise. No extra setup is required for CPU-only use.

## How to use the notebook

1. Activate the virtual environment created above, then launch Jupyter:

   ```bash
   jupyter notebook
   ```

   **or open `autoencoderv3.ipynb` directly in VS Code / JupyterLab / Google Colab**.

2. Run the cells in order from the top:
   - **Imports & pytorch setup**: imports libraries and selects the compute
     device (CUDA/MPS/CPU).
   - **Prepare the dataset**: downloads ChestMNIST automatically on first run
     and builds noisy/clean training, validation and test datasets.
     - Run the "Specialist model" cell to train on one fixed noise
       type/level, OR
     - Run the "Generalist model" cell to train across varied noise
       types/levels.
     - **(Run only one of the two, not both.)**
   - **Convolutional Autoencoder model**: defines the plain CAE and the
     U-Net architectures.
   - **Train the model**: pick a model (CNN-DAE or U-Net), a loss function
     (`mse`, `l1`, `ssim`, `msssim`, `msssim_l1`) and hyperparameters, then run
     the training loop cell. Losses are plotted at the end.
     - An optional "Resume training" section lets you continue training
       from the best saved checkpoint for more epochs.
   - **Test the model & evaluate**: loads the trained checkpoint, computes
     PSNR/SSIM/MSE/MS-SSIM metrics, compares against classical denoising filters, and
     visualizes example outputs and robustness across noise levels.

3. Re-run earlier cells with different parameters (noise type/level,
   model depth, loss function, hyperparameters, etc.) to experiment with different
   configurations.
