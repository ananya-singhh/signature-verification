from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class SignatureDataset(Dataset):
  def __init__(self, df, image_dir = 'data/preprocessed/', size = (256, 512)):
    self.df = df
    self.image_dir = image_dir
    self.size = size

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    transform = transforms.Compose([
        transforms.Resize(self.size),
        transforms.ToTensor(),
    ])

    image0 = Image.open(self.image_dir + self.df.iloc[idx, 0]).convert("L")
    image0 = transform(image0)
    image1 = Image.open(self.image_dir + self.df.iloc[idx, 1]).convert("L")
    image1 = transform(image1)
    label = self.df.iloc[idx, 2]
    writer_id = self.df.iloc[idx, 3]
    signature_id0 = self.df.iloc[idx, 4]
    signature_id1 = self.df.iloc[idx, 5]

    sample = {"image0": image0, "image1": image1, "label": label, "writer_id": writer_id, "signature_id0": signature_id0, "signature_id1": signature_id1}

    return sample