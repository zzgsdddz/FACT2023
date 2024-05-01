from torchvision import datasets, transforms
import argparse
import torch
from torch.utils.data import DataLoader
import clip
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
from torchvision.datasets import CocoDetection

# load the CLIP model
def load_clip_model(config):
    # Splitting the model name from the format "clip:ModelName"
    model_name = config.backbone_name.split(":")[1] if ":" in config.backbone_name else "RN50"
    model, preprocess = clip.load(model_name, device=config.device)
    model.eval()
    return model, preprocess

def load_cifar_dataset(config):
    # Ensuring correct transformation pipeline: From PIL Image to Tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing the image
        transforms.ToTensor(),  # Converting to Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing the image
    ])

    # Load the CIFAR dataset based on the configuration
    if config.dataset == "cifar100":
        dataset = datasets.CIFAR100(root=config.out_dir, train=True, download=True, transform=transform)
    else:  # default to CIFAR10
        dataset = datasets.CIFAR10(root=config.out_dir, train=True, download=True, transform=transform)

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    return loader


COCO_IMAGES_DIR = "../coco/images"
COCO_ANNOTATIONS_DIR = "../coco/annotations/instances_train2017.json"


def load_coco_dataset(config, transform, target_classes):
    dataset = CocoDetection(root=COCO_IMAGES_DIR,
                            annFile=COCO_ANNOTATIONS_DIR,
                            transform=transform)
    def filter_by_class(sample):
        image, target = sample
        labels = [t['category_id'] for t in target]
        label = any(l in target_classes for l in labels)
        return image, torch.tensor([label], dtype=torch.float32)

    filtered_dataset = [filter_by_class(sample) for sample in dataset]
    loader = DataLoader(filtered_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    return loader

class CLIPBinaryClassifier(LightningModule):
    def __init__(self, model, preprocess, lr):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.lr = lr
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model.encode_image(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



# CLIP-based Classifier
class CLIPClassifier(LightningModule):
    def __init__(self, model, preprocess, n_classes, lr):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.n_classes = n_classes
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images):
        if images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)
        image_features = self.model.encode_image(images)
        logits = image_features / torch.norm(image_features, dim=-1, keepdim=True)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone_name", default="clip:RN50")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"], help="Dataset to use (cifar10 or cifar100)")
    args = parser.parse_known_args()[0]
    return args

def run_experiment():
    config = get_config()
    model, preprocess = load_clip_model(config)
    train_loader = load_cifar_dataset(config)
    n_classes = 100 if config.dataset == "cifar100" else 10
    classifier = CLIPClassifier(model, preprocess, n_classes, config.lr)

    trainer = Trainer(
        max_epochs=10,
        accelerator='cpu',
        callbacks=[
            ModelCheckpoint(dirpath='./models/', monitor='train_loss'),
            EarlyStopping(monitor='train_loss', patience=3)
        ]
    )
    trainer.fit(classifier, train_loader)


'''
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--backbone_name", default="ViT-B/32")
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    return args

def run_experiment():
    config = get_config()
    model, preprocess = load_clip_model(config)
    
    transformations = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    # Example target classes (as IDs), adjust as necessary
    target_classes = [3, 6, 8, 17, 20]  # Example class IDs
    train_loader = load_coco_dataset(config, transformations, target_classes)
    
    classifier = CLIPBinaryClassifier(model, preprocess, config.lr)

    trainer = Trainer(
        max_epochs=10,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[
            ModelCheckpoint(dirpath=config.out_dir, monitor='train_loss', mode='min'),
            EarlyStopping(monitor='train_loss', patience=3)
        ]
    )
    trainer.fit(classifier, train_loader)
'''


