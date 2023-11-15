import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class Model(nn.Module):
    def __init__(self, features=None, bins=2, w=0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        # Assume the output of MobileNet features is 576 x 7 x 7 for an input size of 224 x 224 x 3
        self.orientation = nn.Sequential(
            nn.Linear(576 * 7 * 7, 256),  # Adjusted input dimension
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2)
        )
        self.confidence = nn.Sequential(
            nn.Linear(576 * 7 * 7, 256),  # Adjusted input dimension
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
        )
        self.dimension = nn.Sequential(
            nn.Linear(576 * 7 * 7, 512),  # Adjusted input dimension
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 576 * 7 * 7)  # Adjusted flattening
        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension