"""
Date: 13/04/2020
Last Modification: 18/06/2020
Author: Maxime Cauté (maxime.caute@epfl.ch)
---
The classes in this file are used to define our networks.
"""

import torch
import torch.nn as nn

class MultiTriviaNet(nn.Module):
    def __init__(self, channels_amount,image_size, hidden_layers_amount = 1,hidden_layer_size = 80):
        super().__init__()

        self.image_size = image_size
        self.channels_amount = channels_amount

        channel_pixels = image_size**2
        total_pixels = channels_amount*channel_pixels

        self.lin1 = nn.Sequential(
            nn.Linear(total_pixels, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
        )

        self.hids = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size),
                nn.BatchNorm1d(hidden_layer_size),
                nn.ReLU(),
            )
            for i in range(hidden_layers_amount-1)
        ])

        self.lin2 = nn.Sequential(
            nn.Linear(hidden_layer_size, channel_pixels),
            nn.BatchNorm1d(channel_pixels),
        )

    def forward(self, x):

        channel_pixels = self.image_size**2
        total_pixels = self.channels_amount*channel_pixels

        x = x.view(-1, total_pixels)
        x = self.lin1(x)

        #added
        for hidden_layer in self.hids:
            x = hidden_layer(x)

        x = self.lin2(x)
        x = x.view(-1, self.image_size, self.image_size)

        return x



class AtariNet(nn.Module):
    def __init__(self, channels_amount, image_size,
                conv1_channels_amount = 16, conv2_channels_amount = 32,
                conv1_kernel = 8, conv2_kernel = 4,
                conv1_stride = 4, conv2_stride = 2,
                hidden_layer_size = 256
                ):
        super().__init__()

        self.image_size = image_size
        self.channels_amount = channels_amount


        self.conv1 = nn.Sequential(
                nn.Conv2d(self.channels_amount, conv1_channels_amount, conv1_kernel, stride = conv1_stride),
                nn.BatchNorm2d(conv1_channels_amount),
                nn.ReLU(),
            )
        conv1_output_size = int((image_size - conv1_kernel)/conv1_stride) + 1

        self.conv2 = nn.Sequential(
                nn.Conv2d(conv1_channels_amount, conv2_channels_amount, conv2_kernel, stride = conv2_stride),
                nn.BatchNorm2d(conv2_channels_amount),
                nn.ReLU(),
        )
        conv2_output_size = int((conv1_output_size - conv2_kernel)/conv2_stride) + 1

        self.linear_input_size = conv2_channels_amount * conv2_output_size**2
        self.lin1 = nn.Sequential(
            nn.Linear(self.linear_input_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(hidden_layer_size, image_size**2),
            nn.BatchNorm1d(image_size**2),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(-1, self.linear_input_size)
        x = self.lin1(x)
        x = self.lin2(x)
        x = x.view(-1, self.image_size, self.image_size)

        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_size = 30
        self.channels_amount = 2

        channel_pixels = self.image_size**2
        total_pixels = self.channels_amount*channel_pixels

        self.lin1 = nn.Sequential(
            nn.Linear(total_pixels, 2000 ),
            nn.BatchNorm1d(2000),
            #nn.Dropout(0.33), # Limiting a too high weight of 1 neuron
            nn.ReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Linear(2000,channel_pixels),
            nn.BatchNorm1d(channel_pixels),
        )

    def forward(self, x):

        channel_pixels = self.image_size**2
        total_pixels = self.channels_amount*channel_pixels

        x = x.view(-1, total_pixels)
        x = self.lin1(x)
        x = self.lin2(x)
        x = x.view(-1, self.image_size, self.image_size)

        return x
