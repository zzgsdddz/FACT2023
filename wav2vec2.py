from datasets import load_dataset, Audio
from transformers import AutoProcessor, ClapAudioModel, AutoFeatureExtractor

dataset = load_dataset("ashraq/esc50")
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
train_dataset = dataset['train']


# Split the data - 80% for training and 20% for testing
train_data, test_data = dataset.train_test_split(train_dataset, test_size=0.2, random_state=42)
