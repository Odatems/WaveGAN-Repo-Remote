import numpy as np
import torch
from torch import nn
from nn.Self_Attn import Self_Attn

class Generator(nn.Module):

    def __init__(self): 
        
        im_chan=1
        hidden_dim=64
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
           
            nn.ConvTranspose2d(im_chan, im_chan, kernel_size=13, stride=1),
            nn.Conv2d(im_chan,hidden_dim * 4, kernel_size=26,stride=2), # Enas; changed
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            Self_Attn(hidden_dim*2),
            self.make_gen_block(hidden_dim, hidden_dim, kernel_size=4, final_layer=False),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=9,stride=1, final_layer=False),
            nn.Conv2d(im_chan,im_chan,kernel_size = 2, stride=2), # 7,2 if you want 36 
            nn.Sigmoid() 
        )
        

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False): 

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride), 
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Sigmoid(), 
            )

    def forward(self, cost_matrix):
        return self.gen(cost_matrix)

