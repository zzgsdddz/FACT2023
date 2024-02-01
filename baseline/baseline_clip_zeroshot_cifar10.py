import torch
import clip
from torchvision import datasets
from torch.utils.data import DataLoader

class Config:
    def __init__(self):
        self.backbone_name = "clip:RN50"  # CLIP model name
        self.out_dir = "./"  # Data directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64

def load_clip_model(config):
    model, preprocess = clip.load(config.backbone_name.split(":")[1], device=config.device)
    model.eval()
    return model, preprocess

def load_cifar10_dataset(config, preprocess):
    testset = datasets.CIFAR10(root=config.out_dir, train=False, download=True, transform=preprocess)
    test_loader = DataLoader(testset, batch_size=config.batch_size, shuffle=False)
    return test_loader

def zero_shot_classification(model, data_loader, device, class_descriptions):
    model.eval()
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_descriptions]).to(device)
    text_features = model.encode_text(text_inputs)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            image_features = model.encode_image(images)
            logits = image_features @ text_features.T
            predictions = logits.argmax(dim=1)
            correct += (predictions.cpu() == labels).sum().item()
            total += images.shape[0]

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    config = Config()
    clip_model, clip_preprocess = load_clip_model(config)
    cifar10_test_loader = load_cifar10_dataset(config, clip_preprocess)

    # CIFAR-10 Class Descriptions
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # Perform zero-shot classification
    accuracy = zero_shot_classification(clip_model, cifar10_test_loader, config.device, cifar10_classes)
    print(f"Zero-shot classification accuracy on CIFAR10: {accuracy:.2%}")
