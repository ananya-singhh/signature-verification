from models import SiameseNetwork
from dataset import SignatureDataset
import torch
import pandas as pd
from torchvision.models import resnet18
from utils import produce_label

resnet = resnet18()
state_dict = torch.load('resnet/resnet18.pth')
resnet.load_state_dict(state_dict)

model = SiameseNetwork(resnet)
state_dict = torch.load('snn.pth')
model.load_state_dict(state_dict)
model.to('cuda')

df = pd.read_csv('data/images/preprocessed/data.csv')

train_data = SignatureDataset(df, image_dir='data/')

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, pin_memory=True)

for batch_idx, sample in enumerate(train_dataloader):
    print(batch_idx, sample['writer_id'], sample['signature_id1'])
    input1 = sample['image0'].to('cuda')
    input2 = sample['image1'].to('cuda')
    label = sample['label'].to('cuda')

    output1, output2 = model(input1, input2)
    predictions = produce_label(output1, output2, 0.5)

    print(predictions)