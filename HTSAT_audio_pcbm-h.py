import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from transformers import ClapAudioModel, ClapAudioModelWithProjection
from torch.utils.data import Dataset
# from concepts import ConceptBank
from transformers import AutoProcessor, ClapAudioModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricComputer(object):
    def __init__(self, metric_names=None, n_classes=5):
        __all_metrics__ = {"accuracy": self._accuracy, 
                            "class-level-accuracy": self._class_level_accuracy,
                            "confusion_matrix": self._confusion_matrix}
        all_names = list(__all_metrics__.keys())
        if metric_names is None:
            metric_names = all_names
        for n in metric_names: assert n in all_names
        self.metrics = {m: __all_metrics__[m] for m in metric_names}
        self.n_classes = n_classes
    
    def __call__(self, out, target):
        """
        Args:
            out (torch.Tensor): Model output
            target (torch.Tensor): Target labels
        """
        pred = out.argmax(dim=1)
        result = {m: self.metrics[m](out, pred, target) for m in self.metrics.keys()}
        return result
    
    def _accuracy(self, out, pred, target):
        acc = (pred == target).float().detach().mean()
        return acc.item()

    def _class_level_accuracy(self, out, pred, target):
        per_class_acc = {}
        for c in range(self.n_classes):
            count = (target == c).sum().detach().item()
            if count == 0:
                continue
            class_true = ((pred == target) * (target == c)).float().sum().item()
            per_class_acc[c] = (class_true, count)
        return per_class_acc
    
    def _confusion_matrix(self, out, pred, target):
        y_true = target.detach().cpu()
        y_pred = pred.detach().cpu()
        return confusion_matrix(y_true, y_pred, normalize=None, labels=np.arange(self.n_classes))


def unpack_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    elif len(batch) == 2:
        return batch
    else:
        raise ValueError()


@torch.no_grad()
def get_projections(args, backbone, posthoc_layer, loader):
    all_projs, all_embs, all_lbls = None, None, None
    for batch in tqdm(loader):
        batch_X, batch_Y = unpack_batch(batch)
        if "clip" in args.backbone_name:
            embeddings = backbone.encode_image(batch_X).detach().float()
        else:
            embeddings = backbone(**batch_X).audio_embeds.detach()
        projs = posthoc_layer.compute_dist(embeddings).detach().cpu().numpy()
        embeddings = embeddings.detach().cpu().numpy()
        if all_embs is None:
            all_embs = embeddings
            all_projs = projs
            all_lbls = batch_Y.numpy()
        else:
            all_embs = np.concatenate([all_embs, embeddings], axis=0)
            all_projs = np.concatenate([all_projs, projs], axis=0)
            all_lbls = np.concatenate([all_lbls, batch_Y.numpy()], axis=0)
    return all_embs, all_projs, all_lbls


class EmbDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    def __len__(self):
        return len(self.data)


def load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader):
    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    
    # To make it easier to analyize results/rerun with different params, we'll extract the embeddings and save them
    train_file = f"train-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_file = f"test-embs_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_proj_file = f"train-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    test_proj_file = f"test-proj_{args.dataset}__{args.backbone_name}__{conceptbank_source}.npy"
    train_lbls_file = f"train-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    test_lbls_file = f"test-lbls_{args.dataset}__{args.backbone_name}__{conceptbank_source}_lbls.npy"
    

    train_file = os.path.join(args.out_dir, train_file)
    test_file = os.path.join(args.out_dir, test_file)
    train_proj_file = os.path.join(args.out_dir, train_proj_file)
    test_proj_file = os.path.join(args.out_dir, test_proj_file)
    train_lbls_file = os.path.join(args.out_dir, train_lbls_file)
    test_lbls_file = os.path.join(args.out_dir, test_lbls_file)

    if os.path.exists(train_proj_file):
        train_embs = np.load(train_file)
        test_embs = np.load(test_file)
        train_projs = np.load(train_proj_file)
        test_projs = np.load(test_proj_file)
        train_lbls = np.load(train_lbls_file)
        test_lbls = np.load(test_lbls_file)

    else:
        train_embs, train_projs, train_lbls = get_projections(args, backbone, posthoc_layer, train_loader)
        test_embs, test_projs, test_lbls = get_projections(args, backbone, posthoc_layer, test_loader)

        np.save(train_file, train_embs)
        np.save(test_file, test_embs)
        np.save(train_proj_file, train_projs)
        np.save(test_proj_file, test_projs)
        np.save(train_lbls_file, train_lbls)
        np.save(test_lbls_file, test_lbls)
    
    return train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls


class PosthocLinearCBM(nn.Module):
    def __init__(self, concept_bank, backbone_name, idx_to_class=None, n_classes=5):
        """
        PosthocCBM Linear Layer. 
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank)
            backbone_name (str): Name of the backbone, e.g. clip:RN50.
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
        """
        super(PosthocLinearCBM, self).__init__()
        # Get the concept information from the bank
        self.backbone_name = backbone_name
        self.cavs = concept_bank.vectors
        self.intercepts = concept_bank.intercepts
        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()
        self.n_concepts = self.cavs.shape[0]

        self.n_classes = n_classes
        # Will be used to plot classifier weights nicely
        self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

    def compute_dist(self, emb):
        # Computing the geometric margin to the decision boundary specified by CAV.
        margins = (torch.matmul(self.cavs, emb.T) +
           self.intercepts) / (self.norms)
        return margins.T

    def forward(self, emb, return_dist=False):
        x = self.compute_dist(emb)
        out = self.classifier(x)
        if return_dist:
            return out, x
        return out
    
    def forward_projs(self, projs):
        return self.classifier(projs)
    
    def trainable_params(self):
        return self.classifier.parameters()
    
    def classifier_weights(self):
        return self.classifier.weight
    
    def set_weights(self, weights, bias):
        self.classifier.weight.data = torch.tensor(weights).to(self.classifier.weight.device)
        self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
        return 1

    def analyze_classifier(self, k=5, print_lows=False):
        weights = self.classifier.weight.clone().detach()
        output = []

        if len(self.idx_to_class) == 2:
            weights = [weights.squeeze(), weights.squeeze()]
        
        for idx, cls in self.idx_to_class.items():
            cls_weights = weights[idx]
            topk_vals, topk_indices = torch.topk(cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.names[j] for j in topk_indices]
            analysis_str = [f"Class : {cls}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.names[j] for j in topk_indices]
                analysis_str = [f"Class : {cls}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(output)
        return analysis


class PosthocHybridCBM(nn.Module):
    def __init__(self, bottleneck: PosthocLinearCBM):
        """
        PosthocCBM Hybrid Layer. 
        Takes an embedding as the input, outputs class-level predictions.
        Uses both the embedding and the concept predictions.
        Args:
            bottleneck (PosthocLinearCBM): [description]
        """
        super(PosthocHybridCBM, self).__init__()
        # Get the concept information from the bank
        self.bottleneck = bottleneck
        # A single linear layer will be used as the classifier
        self.d_embedding = self.bottleneck.cavs.shape[1]
        self.n_classes = self.bottleneck.n_classes
        self.residual_classifier = nn.Linear(self.d_embedding, self.n_classes)

    def forward(self, emb, return_dist=False):
        x = self.bottleneck.compute_dist(emb)
        out = self.bottleneck.classifier(x) + self.residual_classifier(emb)
        if return_dist:
            return out, x
        return out

    def trainable_params(self):
        return self.residual_classifier.parameters()
    
    def classifier_weights(self):
        return self.residual_classifier.weight

    def analyze_classifier(self):
        return self.bottleneck.analyze_classifier()


class AudioDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

    def __len__(self):
        # Assuming the length of the dataset is the length of 'audio' key
        return len(self.data["audio"])

    def __getitem__(self, idx):
        # Accessing the audio data
        audio_sample = self.data["audio"][idx]["array"]


        # Processing the audio data
        inputs = self.processor(audios=audio_sample, return_tensors="pt")
        inputs["input_features"] = inputs["input_features"][0]

        # Accessing the label/target
        label = self.data["target"][idx]

        return inputs, label
    


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/home/ken/Documents/Uva/Jaar4/FACT/FACT2023-main/pcbm-h-results/", required=False, type=str, help="Output folder")
    parser.add_argument("--pcbm-path", default="/home/ken/Documents/Uva/Jaar4/FACT/FACT2023-main/pcbm_esc-50__HTSAT__clap__lam-1e-05__alpha-0.99__seed-42.ckpt", required=False, type=str, help="Trained PCBM module.")
    parser.add_argument("--concept-bank", default="/home/ken/Documents/Uva/Jaar4/FACT/FACT2023-main/clap.pkl", required=False, type=str, help="Path to the concept bank.")
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset", default="HTSAT", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num-epochs", default=20, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--l2-penalty", default=0.001, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser.parse_args()


@torch.no_grad()
def eval_model(args, posthoc_layer, loader, num_classes):
    epoch_summary = {"Accuracy": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputer(n_classes=num_classes)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
        out = posthoc_layer(batch_X)            
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        metrics = computer(out, batch_Y) 
        epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0]) 
        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        summary_text = "Eval - " + " ".join(summary_text)
        tqdm_loader.set_description(summary_text)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc
    return epoch_summary["Accuracy"]


def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.num_epochs+1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(),
                         "Accuracy": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputer(n_classes=num_classes)
        for batch_X, batch_Y in tqdm(train_loader):
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            cls_loss = cls_criterion(out, batch_Y)
            loss = cls_loss + args.l2_penalty*(posthoc_layer.residual_classifier.weight**2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])
            metrics = computer(out, batch_Y) 
            epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])

            summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
            summary_text = " ".join(summary_text)
            tqdm_loader.set_description(summary_text)
        
        latest_info = dict()
        latest_info["epoch"] = epoch
        latest_info["args"] = args
        latest_info["train_acc"] = epoch_summary["Accuracy"]
        latest_info["test_acc"] = eval_model(args, posthoc_layer, val_loader, num_classes)
        print("Final test acc: ", latest_info["test_acc"])
    return latest_info



def main(args, backbone):

    # Load dataset
    dataset = load_dataset("ashraq/esc50")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    train_dataset = dataset['train']


    # Split the data - 80% for training and 20% for testing
    train_data, test_data = train_test_split(train_dataset, test_size=0.2, random_state=42)

    train_dataset = AudioDataset(train_data, processor)
    test_dataset = AudioDataset(test_data, processor)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Extract class names for each sample
    class_names_per_sample = dataset['train']['category']

    # Extract unique class names and sort them to ensure consistency
    classes = sorted(set(class_names_per_sample))

    # Create a mapping from index to class name
    idx_to_class = {i: class_name for i, class_name in enumerate(classes)}
  
    num_classes = len(classes)
    
    hybrid_model_path = args.pcbm_path.replace("pcbm_", "pcbm-hybrid_")
    run_info_file = hybrid_model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    
    run_info_file = os.path.join(args.out_dir, run_info_file)
    
    # We use the precomputed embeddings and projections.
    train_embs, _, train_lbls, test_embs, _, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)

    
    train_loader = DataLoader(TensorDataset(torch.tensor(train_embs).float(), torch.tensor(train_lbls).long()), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(test_embs).float(), torch.tensor(test_lbls).long()), batch_size=args.batch_size, shuffle=False)

    # Initialize PCBM-h
    hybrid_model = PosthocHybridCBM(posthoc_layer)
    hybrid_model = hybrid_model.to(args.device)
    
    # Initialize the optimizer
    hybrid_optimizer = torch.optim.Adam(hybrid_model.residual_classifier.parameters(), lr=args.lr)
    hybrid_model.residual_classifier = hybrid_model.residual_classifier.float()
    hybrid_model.bottleneck = hybrid_model.bottleneck.float()
    
    # Train PCBM-h
    run_info = train_hybrid(args, train_loader, test_loader, hybrid_model, hybrid_optimizer, num_classes)

    torch.save(hybrid_model, hybrid_model_path)
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)
    
    print(f"Saved to {hybrid_model_path}, {run_info_file}")

if __name__ == "__main__":    
    args = config()    
    # Load the PCBM
    posthoc_layer = torch.load(args.pcbm_path)
    posthoc_layer = posthoc_layer.eval()
    args.backbone_name = posthoc_layer.backbone_name


    # GET BACKBONE
    model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
    backbone = model.to(args.device)
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, backbone)