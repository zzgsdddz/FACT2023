from data.constants import ESC_50, ESC_50_META
from data.esc_50 import load_esc_data
import torch.nn as nn
import torch
import pickle

class ResNet(nn.Module):
    def __init__(self, original_model):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children()))
    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x
    

def train_one_epoch(model, optimizer, training_loader, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss
    

def finetune_resnet(model, args):
    train_loader, _, _ = load_esc_data(ESC_50, ESC_50_META)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for i in range(10):
        print(f"EPOCH: {i}")
        loss = train_one_epoch(model, optimizer, train_loader, loss_fn)
        print(f"loss: {loss}")

    import os
    path = os.path.join(args.out_dir, "resnet18_ESC50.pkl")
    with open(path, "wb") as f:
        pickle.dump(path, f)

    return model