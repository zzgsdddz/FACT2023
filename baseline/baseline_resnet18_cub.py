import torch
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn

class Config:
    def __init__(self):
        self.out_dir = 'C:\\Users\\win11\\PycharmProjects\\post-hoc-cbm\\class_attr_data_10'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 100
        self.num_workers = 4
        self.backbone_name = 'resnet18_cub'

args = Config()

TEST_PKL = os.path.join(args.out_dir, "test.pkl")
TRAIN_PKL = os.path.join(args.out_dir, "train.pkl")
VAL_PKL = os.path.join(args.out_dir, "val.pkl")
CUB_DATA_DIR='C:\\Users\\win11\\PycharmProjects\\post-hoc-cbm\\CUB_200_2011'
class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])
    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x

def load_resnet18_cub(args):
    model = ptcv_get_model(args.backbone_name, pretrained=True, root=args.out_dir)
    model = model.to(args.device)
    model.eval()
    return model

def calculate_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    resnet18_model = load_resnet18_cub(args)
    from cub import load_cub_data

    TEST_PKL = os.path.join(args.out_dir, "test.pkl")

    normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    test_loader = load_cub_data([TEST_PKL], use_attr=False, no_img=False,
                                batch_size=args.batch_size, uncertain_label=False,
                                image_dir=CUB_DATA_DIR, resol=224, normalizer=normalizer,
                                n_classes=200, resampling=True)

    accuracy = calculate_accuracy(resnet18_model, test_loader, args.device)
    print(f'Accuracy of the ResNet18 model on CUB test data: {accuracy:.2%}')



if __name__ == "__main__":
    main()
