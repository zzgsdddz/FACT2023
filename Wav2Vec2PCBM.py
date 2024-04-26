import argparse
import os
import pickle
import numpy as np
import torch
import torch
import torch.nn as nn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import torch
from collections import defaultdict
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoProcessor, ClapAudioModel, AutoFeatureExtractor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import ClapAudioModel, ClapAudioModelWithProjection, AutoModelForAudioClassification

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concept-bank", default="./", required=True, type=str, help="Path to the concept bank")
    parser.add_argument("--out-dir", default="./", required=True, type=str, help="Output folder for model/run info.")
    parser.add_argument("--dataset", default="esc-50", type=str)
    parser.add_argument("--backbone-name", default="Wav2Vec2", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
    parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
    parser.add_argument("--lr", default=1e-3, type=float)
    return parser.parse_args()



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

class ListDataset:
    def __init__(self, images, preprocess=None):
        self.images = images
        self.preprocess = preprocess

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.preprocess:
            image = self.preprocess(image)
        return image


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]
    
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




class AudioDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def __len__(self):
        return len(self.data["audio"])

    def __getitem__(self, idx):
        audio_sample = self.data["audio"][idx]["array"]


        # Processing the audio data
        inputs = self.processor(audios=audio_sample, return_tensors="pt")
        inputs["input_features"] = inputs["input_features"][0]

        # Accessing the label/target
        label = self.data["target"][idx]

        return inputs, label


def run_linear_probe(args, train_data, test_data):
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # We converged to using SGDClassifier. 
    # It's fine to use other modules here, this seemed like the most pedagogical option.
    # We experimented with torch modules etc., and results are mostly parallel.
    classifier = SGDClassifier(random_state=args.seed, loss="log_loss",
                               alpha=args.lam, l1_ratio=args.alpha, verbose=0,
                               penalty="elasticnet", max_iter=10000)
    classifier.fit(train_features, train_labels)

    train_predictions = classifier.predict(train_features)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    predictions = classifier.predict(test_features)
    test_accuracy = np.mean((test_labels == predictions).astype(float)) * 100.

    # Compute class-level accuracies. Can later be used to understand what classes are lacking some concepts.
    cls_acc = {"train": {}, "test": {}}
    for lbl in np.unique(train_labels):
        test_lbl_mask = test_labels == lbl
        train_lbl_mask = train_labels == lbl
        cls_acc["test"][lbl] = np.mean((test_labels[test_lbl_mask] == predictions[test_lbl_mask]).astype(float))
        cls_acc["train"][lbl] = np.mean(
            (train_labels[train_lbl_mask] == train_predictions[train_lbl_mask]).astype(float))
        print(f"{lbl}: {cls_acc['test'][lbl]}")

    run_info = {"train_acc": train_accuracy, "test_acc": test_accuracy,
                "cls_acc": cls_acc,
                }

    # If it's a binary task, we compute auc
    if test_labels.max() == 1:
        run_info["test_auc"] = roc_auc_score(test_labels, classifier.decision_function(test_features))
        run_info["train_auc"] = roc_auc_score(train_labels, classifier.decision_function(train_features))
    return run_info, classifier.coef_, classifier.intercept_



def main(args, concept_bank, backbone):
    dataset = load_dataset("ashraq/esc50", split='train').train_test_split(test_size=0.2)
    processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    train_data = dataset['train']
    test_data = dataset['test']
    train_data = train_data.remove_columns(["filename", "fold", "category", "esc_10", "src_file"])
    print(train_data)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(
            audio_arrays, sampling_rate=processor.sampling_rate, max_length=16000, truncation=True
        )
        return inputs

    train_data = train_data.map(preprocess_function, remove_columns="audio", batched=True)
    test_data = test_data.map(preprocess_function, remove_columns="audio", batched=True)

    # Split the data - 80% for training and 20% for testing
    # print(dataset)
    # train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

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

    # Get a clean conceptbank string
    # e.g. if the path is /../../cub_resnet-cub_0.1_100.pkl, then the conceptbank string is resnet-cub_0.1_100
    # which means a bank learned with 100 samples per concept with C=0.1 regularization parameter for the SVM. 
    # See `learn_concepts_dataset.py` for details.
    conceptbank_source = args.concept_bank.split("/")[-1].split(".")[0] 
    num_classes = len(classes)
    
    # Initialize the PCBM module.
    posthoc_layer = PosthocLinearCBM(concept_bank, backbone_name=args.backbone_name, idx_to_class=idx_to_class, n_classes=num_classes)
    posthoc_layer = posthoc_layer.to(args.device)

    # We compute the projections and save to the output directory. This is to save time in tuning hparams / analyzing projections.
    train_embs, train_projs, train_lbls, test_embs, test_projs, test_lbls = load_or_compute_projections(args, backbone, posthoc_layer, train_loader, test_loader)
    
    run_info, weights, bias = run_linear_probe(args, (train_projs, train_lbls), (test_projs, test_lbls))
    
    # Convert from the SGDClassifier module to PCBM module.
    posthoc_layer.set_weights(weights=weights, bias=bias)

    # Sorry for the model path hack. Probably i'll change this later.
    model_path = os.path.join(args.out_dir,
                              f"pcbm_{args.dataset}__{args.backbone_name}__{conceptbank_source}__lam-{args.lam}__alpha-{args.alpha}__seed-{args.seed}.ckpt")
    torch.save(posthoc_layer, model_path)

    # Again, a sad hack.. Open to suggestions
    run_info_file = model_path.replace("pcbm", "run_info-pcbm")
    run_info_file = run_info_file.replace(".ckpt", ".pkl")
    run_info_file = os.path.join(args.out_dir, run_info_file)
    
    with open(run_info_file, "wb") as f:
        pickle.dump(run_info, f)

    
    if num_classes > 1:
        # Prints the Top-5 Concept Weigths for each class.
        print(posthoc_layer.analyze_classifier(k=5))

    print(f"Model saved to : {model_path}")
    print(run_info)

if __name__ == "__main__":
    args = config()
    all_concepts = pickle.load(open(args.concept_bank, 'rb'))
    all_concept_names = list(all_concepts.keys())
    print(f"Bank path: {args.concept_bank}. {len(all_concept_names)} concepts will be used.")
    concept_bank = ConceptBank(all_concepts, args.device)

    # Get the backbone
    model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base")
    model = model.to(args.device)
    backbone = model
    backbone = backbone.to(args.device)
    backbone.eval()
    main(args, concept_bank, backbone)