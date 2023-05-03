import numpy as np
import torch
from torch import nn
from nn.Self_Attn import Self_Attn

class Critic(nn.Module):    
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            nn.ConvTranspose2d(im_chan, im_chan, kernel_size=13, stride=1),
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            Self_Attn(hidden_dim*4),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):# 
      
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, cost_matrix):
        crit_pred = self.crit(cost_matrix)
        return crit_pred.view(len(crit_pred), -1)
    
    
