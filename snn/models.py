import torchvision
import torch.nn as nn
import torch

class SiameseNetwork(nn.Module):
  def __init__(self, resnet, out_channels=64, out_features=256):
    super(SiameseNetwork, self).__init__()

    # Get ResNet model
    self.resnet = resnet

    # Over-write the 1st conv later to be able to read preprocessed CEDAR images
    # As ResNet reads (3,x,x) where 3 is RGB channels
    # Whereas the data has (1,x,x) where 1 is a gray-scale channel
    self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True)
    self.fc_in_features = self.resnet.fc.in_features

    # Remove the last layer (linear layer before the avgpool one)
    self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
    # maybe freeze some layers

    # # Add linear layers to compare between the features of the two images
    # self.fc = nn.Sequential(
    #     nn.Linear(in_features=self.fc_in_features * 2, out_features=out_features),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(in_features=out_features, out_features=1),
    # )

    # self.sigmoid = nn.Sigmoid()

    # Initialize the weights
    self.fc = nn.Linear(in_features=self.fc_in_features, out_features=out_features)
    self.fc.apply(self.init_weights)

  def init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)

  def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

  def forward(self, input1, input2):
      # Get two images' features
      output1 = self.forward_once(input1)
      output2 = self.forward_once(input2)

      # # Concatenate both images' features
      # output = torch.cat((output1, output2), 1)

      # # Pass the concatenation to the linear layers
      # output = self.fc(output)

      # # Pass the output of the linear layers to sigmoid layer
      # output = self.sigmoid(output)

      return output1, output2