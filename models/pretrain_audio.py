from data.constants import ESC_50, ESC_50_META
from data.esc_50 import load_esc_data
import torch.nn as nn
import torch
import pickle
import tqdm

class ResNet(nn.Module):
    def __init__(self, original_model):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children()))
    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x
    

def train_one_epoch(model, optimizer, train_loader, loss_fn, device, epochs=50, change_lr=None):
# def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
    for epoch in tqdm(range(1,epochs+1)):
        model.train()
        batch_losses=[]
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
            # train_losses.append(batch_losses)
            # print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])}')

    
def eval(model, validation_loader, device):
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for i, vdata in enumerate(validation_loader):
            x, y = vdata
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    print(f"test_acc: {float(num_correct)/float(num_samples)*100:.2f}")


def finetune_resnet(model, args):
    train_loader, test_loader, _ = load_esc_data(ESC_50, ESC_50_META)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for i in range(10):
        print(f"EPOCH: {i}")
        loss = train_one_epoch(model, optimizer, train_loader, loss_fn)
        print(f"loss: {loss}")

    eval(model, test_loader, args.device)

    import os
    path = os.path.join(args.out_dir, "resnet18_ESC50.pth")
    torch.save(model.state_dict(), path)

    return model