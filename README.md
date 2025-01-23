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
```


## Training
The training pipeline is implemented in the scripts/train.py file. The script loads the dataset, creates the training and validation DataLoader instances, and trains the U-Net model for the specified number of epochs.

1. Training Procedure
2. The dataset is split into training and validation sets.
3. The model is trained using the Adam optimizer with a learning rate of 1e-3.
4. For each batch, the model computes a loss using partial cross-entropy, focusing on labeled pixels.
5. The model is evaluated after each epoch on the validation set.
6. The model's state is saved after training completes.
7. Example Command to Train the Model
   
To start training, run the following command:

```python
scripts/train.py
```

The model is saved at the end of training as unet_model.pth.

## Results
Once the model has been trained, you can evaluate its performance using various metrics such as loss and Intersection over Union (IoU). The performance is reported both during training and validation.

## Metrics
1. Loss: Cross-entropy loss on the labeled pixels.
2. IoU: Intersection over Union score, used to evaluate the segmentation accuracy.
