# UNet-Remote-Sensing-Image-Segmentation-for-Early-Fire-Detection

This project implements a deep learning model using U-Net architecture for remote sensing image segmentation, aimed at early fire detection. It leverages a pretrained ResNet34 backbone for feature extraction, and it is designed to detect areas where fires are beginning by analyzing satellite images.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Overview

The goal of this project is to train a U-Net model to segment satellite images and detect fire-prone areas. The model uses a ResNet34 backbone pretrained on ImageNet for feature extraction, which is fine-tuned for the task of detecting early fire signs in satellite images.

## Dependencies

### Python Packages

This project requires the following Python packages:

- `numpy`: For numerical operations.
- `matplotlib`: For visualizing data and results.
- `tqdm`: For progress bars during training and evaluation.
- `scikit-learn`: For machine learning tools.
- `torch`: For PyTorch deep learning framework.
- `torchvision`: For image transformations and vision-based operations.
- `segmentation-models-pytorch`: For pre-implemented segmentation models including U-Net.
- `datasets`: For loading and processing datasets from the Hugging Face hub.

These dependencies are listed in the `environment.yml` and `requirements.txt` files.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your_username/yekhanfir-early-fire-detection-from-satelite-images-with-u-net.git
    ```

2. Navigate into the project directory:

    ```bash
    cd yekhanfir-early-fire-detection-from-satelite-images-with-u-net
    ```

3. Create a conda environment using `environment.yml`:

    ```bash
    conda env create -f environment.yml
    ```

4. Activate the environment:

    ```bash
    conda activate remote-sensing-segmentation
    ```

5. Install additional dependencies from `requirements.txt` using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used in this project is based on the California Burned Areas dataset, specifically the post-fire imagery. The dataset is loaded from the Hugging Face `datasets` library using the following command:

```python
dataset_post_fire = load_dataset("DarthReca/california_burned_areas", name="post-fire")
