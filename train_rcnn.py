import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import Resize

from custom_dataset import CustomDataset
from config import Config

def collate_fn(batch):
    # Filter out any samples where `__getitem__` returned None
    batch = [sample for sample in batch if sample is not None]
    
    # If the filtered batch size is less than the expected batch size, discard it
    if len(batch) < 2:  # or whatever batch size you expect
        return None  # Skip this batch if it doesn't meet the requirement

    # Otherwise, return the batch with images and targets separated
    return tuple(zip(*batch))

import time
if __name__ == "__main__":
    t = time.time()
    
    config = Config()
    
    # Transformations
    transform = T.Compose([
        Resize((256, 256)),  # Resize to 256x256 pixels
        T.ToTensor(),
    ])
    

    # Datasets and DataLoaders
    train_dataset = CustomDataset(image_dir="dataset/train/images", annot_dir="dataset/train/annotations", transforms=transform)
    valid_dataset = CustomDataset(image_dir="dataset/valid/images", annot_dir="dataset/valid/annotations", transforms=transform)
    test_dataset = CustomDataset(image_dir="dataset/test/images", annot_dir="dataset/test/annotations", transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False,num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,num_workers=8)

    # Load the Faster R-CNN model with ResNet-50
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    num_classes = len(config.LABELS) + 1  # Plus one for the background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Move the model to GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    model.to(device)

    # Training function
    def train_one_epoch(model, optimizer, data_loader, device):
        model.train()
        total_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # Calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        return total_loss / len(data_loader)

    # Evaluation function
    @torch.no_grad()
    def evaluate(model, data_loader, device):
        model.eval()
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # Perform evaluation metrics if necessary

    # Main training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        
        print(f"Training epoch {epoch}, time = {time.time()-t}")
        
        train_loss = train_one_epoch(model, optimizer, train_loader, device)

        print(f"Finished training epoch {epoch}/{num_epochs}, ")
        # Validate on validation set
        evaluate(model, valid_loader, device)
        print(f"epoch {epoch}/{num_epochs}, {time.time()-t}s")
    model._save_to_state_dict("model.pth")
    print("Training completed in ", time.time()-t)
