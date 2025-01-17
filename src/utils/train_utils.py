import torch
import torch.nn as nn

def partial_cross_entropy_loss(predictions, labels, labeled_mask):
    labeled_predictions = predictions[labeled_mask]
    labeled_labels = labels[labeled_mask]
    if labeled_predictions.size(0) == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(labeled_predictions, labeled_labels)

def calculate_iou(predictions, targets, threshold=0.5):
    predictions = (torch.sigmoid(predictions) > threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = (predictions + targets).sum(dim=(1, 2, 3)) - intersection
    return (intersection / (union + 1e-6)).mean().item()