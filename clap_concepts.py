import requests
import os
import pickle
import torch
import clip
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, ClapModel
from numpy import genfromtxt



def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=False, default="/home/ken/Documents/Uva/Jaar4/FACT/clap_out", type=str)
    parser.add_argument("--classes", default="esc50", type=str)
    parser.add_argument("--backbone-name", default="clip:RN50", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--recurse", default=1, type=int, help="How many times to recurse on the conceptnet graph")
    return parser.parse_args()


def get_single_concept_data(cls_name):
    if cls_name in concept_cache:
        return concept_cache[cls_name]
    
    all_concepts = []
    
    # Has relations
    has_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasA&start=/c/en/{}"
    obj = requests.get(has_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Made of relations
    madeof_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/MadeOf&start=/c/en/{}"
    obj = requests.get(madeof_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Properties of things
    property_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/HasProperty&start=/c/en/{}"
    obj = requests.get(property_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['end']['label'])
    
    # Categorization concepts
    is_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/IsA&start=/c/en/{}"
    obj = requests.get(is_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        if edge["weight"] <= 1:
            continue
        all_concepts.append(edge['end']['label'])
    
    # Parts of things
    parts_query = "https://api.conceptnet.io/query?node=/c/en/{}&rel=/r/PartOf&end=/c/en/{}"
    obj = requests.get(parts_query.format(cls_name, cls_name)).json()
    for edge in obj["edges"]:
        all_concepts.append(edge['start']['label'])
    
    all_concepts = [c.lower() for c in all_concepts]
    # Drop the "a " for concepts defined like "a {concept}".
    all_concepts = [c.replace("a ", "") for c in all_concepts]
    # Drop all empty concepts.
    all_concepts = [c for c in all_concepts if c!=""]
    # Make each concept unique in the set.
    all_concepts = set(all_concepts)
    
    concept_cache[cls_name] = all_concepts
    
    return all_concepts


def get_concept_data(all_classes):
    all_concepts = set()
    # Collect concepts that are relevant to each class
    for cls_name in all_classes:
        print(f"Pulling concepts for {cls_name}")
        all_concepts |= get_single_concept_data(cls_name)
    return all_concepts


def get_concept_data2(all_classes):
    import pandas as pd
    all_concepts = set()
    # Collect concepts that are relevant to each class
    # data = genfromtxt('/home/ken/Documents/Uva/Jaar4/FACT/clap_gpt_concept.csv', delimiter=',')

    df = pd.read_csv('C:/Users/lenna/Documents/UvA/FACT/post-hoc-cbm/data/clap_gpt_concepts_20each.csv', sep=',', header=None)
    data = df.to_numpy().flatten()

    print(len(data))
    print(data)
    
    return set(data)


def clean_concepts(scenario_concepts):
    """
    Clean the plurals, trailing whitespaces etc.
    """
    from nltk.stem.wordnet import WordNetLemmatizer
    import nltk

    # We use nltk to handle plurals, multiples of the same words etc.
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    Lem = WordNetLemmatizer()

    scenario_concepts_rec = []
    for c_prev in scenario_concepts:
        c = c_prev
        if type(c) != str: continue
        c = c.strip()
        c_subwords = c.split(" ")
        # If a concept is made of more than 2 words, we drop it.
        if len(c_subwords) > 2:
            print("skipping long concept", c_prev)
            continue
        # Lemmatize words to help eliminate non-unique concepts etc.
        for i, csw in enumerate(c_subwords):
            c_subwords[i] = Lem.lemmatize(csw)
        lemword = " ".join(c_subwords)
        if c_prev == lemword:
            scenario_concepts_rec.append(c)
        else:
            if lemword in scenario_concepts:
                print(c, lemword)
            else:
                scenario_concepts_rec.append(c)
    scenario_concepts_rec = list(set(scenario_concepts_rec))
    return scenario_concepts_rec


@torch.no_grad()
def learn_conceptbank(args, concept_list, scenario):
    concept_dict = {}
    for concept in tqdm(concept_list):
        # Note: You can try other forms of prompting, e.g. "photo of {concept}" etc. here.

        # Get text features 

        # text = clip.tokenize(f"{concept}").to("cuda")
        # text_features = model.encode_text(text).cpu().numpy()
        # text_features = text_features/np.linalg.norm(text_features)

        # Assuming 'concept' is your text input and 'model' and 'tokenizer' are already defined as shown previously
        inputs = tokenizer(concept, padding=True, return_tensors="pt").to("cpu")
        text_features = model.get_text_features(**inputs).cpu().numpy()
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)


        # store concept vectors in a dictionary. Adding the additional terms to be consistent with the
        # `ConceptBank` class (see `concepts/concept_utils.py`).
        concept_dict[concept] = (text_features, None, None, 0, {})

    print(f"# concepts: {len(concept_dict)}")
    if args.backbone_name == 'clip:RN50':
        concept_dict_path = os.path.join(args.out_dir, f"multimodal_concept_clap_{scenario}_recurse-{args.recurse}.pkl")
    else:
        concept_dict_path = os.path.join(args.out_dir, f"multimodal_concept_{args.backbone_name}_{scenario}_recurse-{args.recurse}.pkl")
    pickle.dump(concept_dict, open(concept_dict_path, 'wb'))
    print(f"Dumped to : {concept_dict_path}")


if __name__ == "__main__":
    args = config()

    # Loading Clap model and tokenizer
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")

    tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    # model, _ = clip.load(args.backbone_name.split(":")[1], device=args.device, download_root=args.out_dir)
    concept_cache = {}
    
    if args.classes == "esc50":
        # Pull esc50 to get the class names.
        print("loading dataset")
        dataset = load_dataset("ashraq/esc50")

        audio_sample = dataset["train"]["audio"][0]["array"]
        # Assuming the dataset is already loaded and 'train' split is available
        unique_categories = set()

        # Iterate over all rows in the 'train' split to extract unique category names
        for sample in dataset['train']:
            unique_categories.add(sample['category'])

        # Convert the set to a list if you need it in list format
        all_classes = list(unique_categories)
        print(len(all_classes))
        assert(all_classes != 50)
        print("passed assert")
        # Get the names of all concepts.
        all_concepts = get_concept_data2(all_classes)
        # Clean the concepts for uniques, plurals etc. 
        all_concepts = clean_concepts(all_concepts)     
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data2(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # Generate the concept bank.
        learn_conceptbank(args, all_concepts, args.classes)
        
    elif args.classes == "cifar100":
        from torchvision import datasets
        cifar100_ds = datasets.CIFAR100(root=args.out_dir, train=True, download=True)
        all_classes = list(cifar100_ds.classes)
        all_concepts = get_concept_data(all_classes)
        all_concepts = clean_concepts(all_concepts)
        all_concepts = list(set(all_concepts).difference(set(all_classes)))
        # If we'd like to recurse in the conceptnet graph, specify `recurse > 1`.
        for i in range(1, args.recurse):
            all_concepts = get_concept_data(all_concepts)
            all_concepts = list(set(all_concepts))
            all_concepts = clean_concepts(all_concepts)
            all_concepts = list(set(all_concepts).difference(set(all_classes)))
        learn_conceptbank(args, all_concepts, args.classes)

    else:
        raise ValueError(f"Unknown classes: {args.classes}. Define your dataset here!")