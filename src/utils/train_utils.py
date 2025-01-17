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

def train_model(model, train_loader, val_loader, device, epochs, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        train_loss, train_iou = 0.0, 0.0
        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images = images.permute(0, 3, 1, 2).to(device)
            masks = masks.to(device)
            outputs = model(images)
            labeled_mask = masks != -1
            loss = partial_cross_entropy_loss(outputs.squeeze(1).float(), masks.float(), labeled_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += calculate_iou(outputs, masks.unsqueeze(1))
        print(f"Epoch {epoch+1}: Loss = {train_loss/len(train_loader):.4f}, IoU = {train_iou/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.permute(0, 3, 1, 2).to(device)
                masks = masks.to(device)
                outputs = model(images)
                labeled_mask = masks != -1
                loss = partial_cross_entropy_loss(outputs.squeeze(1).float(), masks.float(), labeled_mask)
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks.unsqueeze(1))
        print(f"Validation: Loss = {val_loss/len(val_loader):.4f}, IoU = {val_iou/len(val_loader):.4f}")
        
    torch.save(model.state_dict(), save_path)