import os
import torch
from torch.utils.data import DataLoader
from src.models.unet_model import get_unet_model
from src.utils.dataset import SegmentationDataset
from src.utils.train_utils import train_model, validate_model
from datasets import load_dataset

# Load the dataset
dataset_post_fire = load_dataset("DarthReca/california_burned_areas", name="post-fire")

# Select training and validation folds
train_folds = list(dataset_post_fire.keys())[:-1]
validation_folds = [list(dataset_post_fire.keys())[-1]]

# Create dataset and dataloader
train_dataset = SegmentationDataset(dataset_post_fire, train_folds)
validation_dataset = SegmentationDataset(dataset_post_fire, validation_folds)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_unet_model().to(device)

# Train the model
train_model(
    model=model,
    train_loader=train_dataloader,
    val_loader=validation_dataloader,
    device=device,
    epochs=10,
    save_path="unet_model.pth"
)
