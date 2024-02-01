import torch
import os
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np

# Add your imports for derma_models and derma_data or equivalent modules
from derma_models import get_derma_model
from derma_data import load_ham_data


class Config:
    def __init__(self):
        self.out_dir = 'C:\\Users\\win11\\PycharmProjects\\post-hoc-cbm\\class_attr_data_10'
        self.device = 'cpu'
        self.batch_size = 100
        self.num_workers = 4
        self.backbone_name = 'ham10000_inception'
        self.seed = 42


def calculate_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    args = Config()
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load data
    train_loader, test_loader, idx_to_class = load_ham_data(args, preprocess)

    # Load model
    model, backbone, model_top = get_derma_model(args, args.backbone_name)
    model = model.to(args.device)
    model.eval()

    # Calculate accuracy
    accuracy = calculate_accuracy(model, test_loader, args.device)
    print(f'Accuracy on test set: {accuracy:.2%}')

