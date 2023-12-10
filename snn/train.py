import torch
from models import SiameseNetwork
from dataset import SignatureDataset
from losses import ContrastiveLoss
from torch.optim import lr_scheduler
import torch.optim as optim
import pandas as pd
from utils import fit
from torchvision.models import resnet18

df = pd.read_csv('data/data_new.csv')#, nrows=2000)
train_df = df[df['writer_id'] <= 44]
test_df = df[df['writer_id'] > 44]

train_data = SignatureDataset(train_df)
test_data = SignatureDataset(test_df)

# 2, 2, 2, 3, 5, 11, 71
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

margin = 1.

# resnet18 = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V2")
model = resnet18()
state_dict = torch.load('resnet/resnet18.pth')
model.load_state_dict(state_dict)
# resnet = {18: resnet18(weights=ResNet18_Weights.DEFAULT)}
resnet = {18: model}

with open('terminal.txt', 'w') as f:
    for resnet_layers in resnet.keys():
        for lr in [1e-5, 1e-4, 1e-3]:
            for out_features in [128, 256, 512]:
                for threshold in [0.4, 0.5, 0.6]:
                    model = SiameseNetwork(resnet[resnet_layers], out_features=out_features)
                    model.to('cuda')
                    criterion = ContrastiveLoss(margin)
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
                    n_epochs = 1
                    log_interval = 1
                    log_dir = f'experiment_resnet{resnet_layers}_lr{lr}_outfeatures{out_features}_threshold{threshold}'

                    print(log_dir, file=f)

                    fit(train_dataloader, test_dataloader, model, criterion, optimizer, scheduler, n_epochs, False, log_interval, 0.5, log_dir, f)

                    torch.save(model.state_dict(), log_dir + '_snn.pth')

                    print('\n', file=f, flush=True)

f.close()