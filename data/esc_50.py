import librosa
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from tqdm import tqdm

preprocess = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),   # go from 1 channel to 3
            # transforms.Lambda(lambda x: torch.squeeze(x,0)),
            transforms.Resize((224,224)),
            transforms.CenterCrop((224,224)),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
  spec_scaled = preprocess(spec_scaled)
  return spec_scaled


def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
  wav,sr = librosa.load(file_path,sr=sr)
  if wav.shape[0]<5*sr:
    wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
  else:
    wav=wav[:5*sr]
  spec=librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
  spec_db=librosa.power_to_db(spec,top_db=top_db)
  return spec_db


class ESC50Data(Dataset):
  def __init__(self, base, df, in_col, out_col):
    self.df = df
    self.data = []
    self.labels = []
    self.c2i={}
    self.i2c={}
    self.categories = sorted(df[out_col].unique())
    for i, category in enumerate(self.categories):
      self.c2i[category]=i
      self.i2c[i]=category
    for ind in tqdm(range(len(df))):
      row = df.iloc[ind]
      file_path = os.path.join(base,row[in_col])
      self.data.append(spec_to_image(get_melspectrogram_db(file_path)))#[np.newaxis,...])
      self.labels.append(self.c2i[row['category']])
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    return self.data[idx], self.labels[idx]
  

def load_esc_data(ESC_50, ESC_50_META):
    df = pd.read_csv(os.path.join(ESC_50_META, 'esc50.csv'))
    train = df[df['fold']!=5]
    test = df[df['fold']==5] 

    train_data = ESC50Data(os.path.join(ESC_50, 'audio'), train, 'filename', 'category')


    test_data = ESC50Data(os.path.join(ESC_50, 'audio'), test, 'filename', 'category')


    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    idx_to_class = {v: k for k, v in train_data.c2i.items()}

    return train_loader, test_loader, idx_to_class

