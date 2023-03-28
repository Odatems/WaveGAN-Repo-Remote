import torch
from torch import nn
#from tqdm.auto import tqdm
#from torchvision import transforms
#from torchvision.datasets import MNIST
from torchvision.utils import make_grid
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

padding=(3,4,3,4)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook

#-----------------------------------------------------------------------
# This part is to add attention layer 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()
        
        # Construct the conv layers
        self.query_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim//2 , kernel_size= 1)
        
        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        
        '''
        print(C)
        print(width)
        print(height)
        '''

        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B * N * C
        #print(proj_query.size)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        energy =  torch.bmm(proj_query, proj_key) # batch matrix-matrix product
        #print(m_batchsize)
        
        attention = self.softmax(energy) # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        out = torch.bmm(proj_value, attention.permute(0,2,1)) # batch matrix-matrix product
        out = out.view(m_batchsize,C,width,height) # B * C * W * H
        
        # Add attention weights onto input
        out = self.gamma*out + x
        #print(out.shape)
        return out #, attention


#-------------------------------------------------------------------------

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self): # Enas, removed  z_dim=10,
        im_chan=1
        hidden_dim=64
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
           #self.make_gen_block(im_chan, im_chan , kernel_size=14, stride=1),
            nn.ConvTranspose2d(im_chan, im_chan, kernel_size=13, stride=1),
            #nn.ConstantPad2d(padding, value=0),
            #nn.ZeroPad2d(((11, 12),(11,12))), # Enas: added to resize the input image
            nn.Conv2d(im_chan,hidden_dim * 4, kernel_size=26,stride=2), # Enas; changed
            #self.make_gen_block(z_dim, hidden_dim * 4), 
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            Self_Attn(hidden_dim*2),
            self.make_gen_block(hidden_dim, hidden_dim, kernel_size=4, final_layer=False),
            #self.make_gen_block(hidden_dim, hidden_dim, kernel_size=3, stride= 1,final_layer=False),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=9,stride=1, final_layer=False),
            #nn.Linear(im_chan,5,bias=False), # Enas: Added for sizing issue. You may consider adding con2d layer instead.            
            #nn.Conv2d(im_chan,im_chan, kernel_size=7, stride=1), # This is the first convolution layer # Enas; changed was kernel_size=20 in case of 5, kernel_size=14 in case of 8, 10 in 10
            #nn.Conv2d(im_chan,im_chan,kernel_size = 4, stride=2), # second convolution layer 
            nn.Conv2d(im_chan,im_chan,kernel_size = 2, stride=2), # 7,2 if you want 36 
            nn.Sigmoid() # added to cancel the effect of teh negative values
            
            
        )
        

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False): 
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride), #nn.utils.spectral_norm(
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                #nn.Tanh(),
                #nn.Conv2d(im_chan,hidden_dim * 4, kernel_size=26),
                nn.Sigmoid(), # Enas, added by me to check the results 
            )

    def forward(self, cost_matrix): # Enas: changed. was:   noise
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        #x = noise.view(len(noise), self.z_dim, 1, 1) # Enas, changed
        #image_height, image_width = 28
        #print('this point')
        # cost_matrix = cost_matrix.view(batch_size, image_height,image_width, 1) # commented tried to apply it 
        #x = cost_matrix 
        # cost_mat = self.gen(cost_matrix) # step 1 in size adjustment 
        # cost_mat = cost_mat.view(batch_size,1,15,15) # step 2 in code adjustment
        # return cost_mat#self.gen(cost_mat)#.view(1,1,15,15) # step 3 in code adjustment
        #print(cost_mat.shape)
        return self.gen(cost_matrix)#.view(1,1,15,15)

def get_noise(n_samples, z_dim, device='cpu'): # Enas, marked as unnecesary. Done. 
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

# We define a generator with latent dimension 100 and img_dim 1
# In this code I wanted to test the noie vector.
"""
gen = Generator(100)
print("Composition of the Generator:", end="\n\n")
print(gen)

z_noise = get_noise(1, 100, device='cpu')
z_noise
g = gen(z_noise)
g.shape

"""




# ------------------- here we are generating and preparing the data -------------------------------------------
# This is the code without duplication of dataset 

import time
import argparse
import pprint as pp
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from Gurobi_tsp_reader import GurobiTSPReader

dataset_size_all = 546573 #546000
image_height,image_width = 20,20
num_nodes = 20
num_nodes = image_height 
validation_set_size = 32 # This is the size of data used for validation during training 

rep_factor = 1 #10
counter = 0

xx = np.zeros((dataset_size_all * rep_factor,1,num_nodes,num_nodes)) # changed to mtch pytorch 
org_cost = np.zeros((dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # changed to match pytorch 
z_norm = np.zeros((dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # changed to match pytorch 
optimal_cost = np.zeros((dataset_size_all * rep_factor,1,num_nodes,num_nodes))


num_neighbors= -1
batch_size= 1 # Warning =;;;;;; next you may not use it s one .
train_filepath = f"mmwave20_gurobi_multi_proc.txt" #f"mmwave{num_nodes}_train_Gurobi.txt"
dataset = GurobiTSPReader(num_nodes, num_neighbors, batch_size, train_filepath)
print("Number of batches of size {}: {}".format(batch_size, dataset.max_iter))
batch = next(iter(dataset))  # Generate a batch of TSPs


start_time = time.time()
  
i = iter(dataset)


for itr_num in range(np.int32(dataset_size_all)): # in case of repetition dataset_size_all/rep_factor
    
    next_batch = next(i)
    #Cost = np.multiply(next_batch.edges_values,next_batch.edges_target)
    xx[itr_num] = next_batch.edges_target
    #xx[itr_num] = Cost
    # In case that you want the ful matrix 
    z_norm[itr_num] = next_batch.edges_values
    
    # Get the optimal tour
    # Get the tour nodes 
    
      
plt.imshow(xx[3,0,:,:])
plt.show()

plt.imshow(z_norm[3,0,:,:])
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))



# -----------prepare validation dataset for training-----------------------



valid_dataset_size_all = 10000
validation_set_sample = np.zeros((valid_dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # take slice of the dataset to test on 

org_cost = np.zeros((valid_dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # changed to match pytorch 
z_norm_valid = np.zeros((valid_dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # changed to match pytorch 

validation_set_sample = xx[500198:510198,:,:]
z_norm_valid = z_norm[500198:510198,:,:]

'''
counter = 0

#num_neighbors= -1
#batch_size= 1 # Warning =;;;;;; next you may not use it s one .
train_filepath =  f"mmwave20_val_Gurobi_multi_proc.txt"#f"mmwave{num_nodes}_val_Gurobi.txt"
#dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size, train_filepath)
#print("Number of batches of size {}: {}".format(batch_size, dataset.max_iter))
#batch = next(iter(dataset))  # Generate a batch of TSPs
start_time = time.time()
i = iter(dataset)

valid_itr_num = 0
for itr_num in range(np.int32(valid_dataset_size_all)):# in case of repetition dataset_size_all/rep_factor
    
    
    next_batch = next(i)
    #Cost = np.multiply(next_batch.edges_values,next_batch.edges_target)
    # In case you want the full matrix 
    validation_set_sample[valid_itr_num] = next_batch.edges_target
 
    #validation_set_sample[itr_num] = Cost
    
    # In case that you want the ful matrix 
    z_norm_valid[valid_itr_num] = next_batch.edges_values
    
    valid_itr_num = valid_itr_num + 1
    # Get the optimal tour
    #optimal_tour_len[itr_num]=next_batch.tour_len
    #optimal_tour_nodes[itr_num] = next_batch.tour_nodes
'''
#--------------------------------------------------------------------------------------------------------------------------------------

# inp[0,0,:,:] - z_norm[0,0,:,:] zero difference to the order of 10^-9 and less

class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    
    '''
    #padding = (11,12,11,12) # left, right, upper, lower. Note that this is cutomized to graph size of 5
    # nn.ConstantPad2d(padding, value=0),
    #nn.ZeroPad2d(((11, 12),(11,12))), # addded to resize the ipt image
    self.make_crit_block(im_chan, hidden_dim),
    self.make_crit_block(hidden_dim, hidden_dim * 2, kernel_size=3),# kernel changed to 3 in case of 16
    #Self_Attn(hidden_dim*4),# this is added to conside rthe attention in the generator 
    self.make_crit_block(hidden_dim * 2, 1, kernel_size=2, final_layer=True), # kernel size changed to 2 in case of 16
    '''
    
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            
           #padding = (11,12,11,12) # left, right, upper, lower. Note that this is cutomized to graph size of 5
            #
            #nn.ConstantPad2d(padding, value=0),
            nn.ConvTranspose2d(im_chan, im_chan, kernel_size=13, stride=1),
            #nn.ZeroPad2d(((11, 12),(11,12))), # addded to resize the ipt image
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            Self_Attn(hidden_dim*4), # addded to considee the attention
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
            
            
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):# 
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride), #nn.utils.spectral_norm(nn.Conv2d(input_channels, output_channels, kernel_size, stride)),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)

n_epochs = 50
z_dim = 64
display_step = 500
batch_size = 32
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda'

# Enas: changed to commented. This is not needed function

gen = Generator().to(device) # Enas: I changed it so it is not taking anything now. it was z_dim

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device) 
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gradient
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    
    #m = nn.ConstantPad2d(padding, value=0)
    #real_pad = m(real)

    #plt.imshow(real_pad[0,0,:,:].detach().cpu().numpy())
    #plt.show()

    #print(real_pad[0,0,:,:].detach().cpu().numpy())
    #print(real.shape)
    #print(real_pad.shape)
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: gradient_penalty
def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(crit_fake_pred, fake, real,g_on_real_pred,epoch_num):#, d_on_real_pred identity loss
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    lambda_recon = 200
    recon_criterion =    nn.L1Loss() #nn.MSELoss()
    gen_rec_loss = recon_criterion(real, fake)
    
    recon_criterion_iden =  nn.L1Loss() 
    identity_loss = recon_criterion_iden(real, g_on_real_pred)
    
    
    '''
    lambda_recon = 200
    recon_criterion = nn.MSELoss()#nn.L1Loss() 
    gen_rec_loss = recon_criterion(real, image_to_handle_thresh_dev)
    '''
    
    #print(fake)
    #print(real)
    gen_loss = -1. * torch.mean(crit_fake_pred) + (lambda_recon) * gen_rec_loss+ identity_loss #lambda_recon*(1/epoch_num) * gen_rec_loss #+ 200*identity_loss

    #### END CODE HERE ####
    return gen_loss#, gen_rec_loss

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_crit_loss
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):#, crit_fake_on_real_pred in case identity loss
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp #+ torch.mean(crit_real_pred) - torch.mean(crit_fake_on_real_pred) 
    #### END CODE HERE ####
    return crit_loss

'''
padding = (2,1,4,3) # left, right, upper, lower
m = nn.ConstantPad2d(padding, value=0)
input = torch.randn( 3, 3)
m(input)
'''

#len(batch_index)

import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
from pandas.compat import numpy
import numpy as np
import pandas as pd



# Prepare the epochs for training 
dataset_size_train_split = dataset_size_all # was 255 in the case of one graph repeated 
X_train = xx
batch_index = [n for n in range(0, dataset_size_train_split, batch_size)] # divide the the range of the dataset size into batches to use them to index the training over batches 
#print(batch_index)
end_index =  0 # before strting the training it should start from 0
#print(range(len(batch_index)))


cur_step = 0
generator_losses = []
critic_losses = []
mse_all = np.zeros((5*n_epochs, 1)) #np.zeros((n_epochs, len(batch_index), validation_set_size+1))
mse_each = np.zeros((5*n_epochs, valid_dataset_size_all))


PATH_G = "WGAN_model_Gen.pt" # specify the location to save the model parameters.
PATH_D = "WGAN_model_Desc.pt"

reconst_loss = []
generator_losses_epoch=[]
reconst_loss_epoch =[]
critic_losses_epoch = []

# load the saved checkpoints
# First: load for the generator

PATH_G_Best = "WGAN_model_Gen_best.pt" # specify the location to save the model parameters.
PATH_D_Best = "WGAN_model_Desc_best.pt"

'''
checkpoint_G = torch.load(PATH_G)
gen.load_state_dict(checkpoint_G['model_state_dict'])
gen_opt.load_state_dict(checkpoint_G['optimizer_state_dict'])
epoch = checkpoint_G['epoch']
generator_losses = checkpoint_G['Loss']

# Second: Load for descriminator
checkpoint_D = torch.load(PATH_D)
crit.load_state_dict(checkpoint_D['model_state_dict'])
crit_opt.load_state_dict(checkpoint_D['optimizer_state_dict'])
epoch = checkpoint_D['epoch']
critic_losses = checkpoint_D['Loss']
'''

mse_long_term_training = np.zeros((10*n_epochs, 1))
#mse_long_term_training[:n_epochs] = mse_all

mse_each_long_term_training = np.zeros((10*n_epochs, valid_dataset_size_all))

mse_best = 100#0.09996766 #0.05353824 #100

for epoch in range(10*n_epochs):#,2*n_epochs

    torch.cuda.empty_cache()

    end_index = 0
    start_index = end_index # find the starting point of the current batch
    
    # Dataloader returns the batches
    for iteration in range(len(batch_index)-1): # for real, _ in tqdm(dataloader):# Enas: changed 

        start_index = end_index # find the starting point of the current batch
        end_index = start_index + batch_size # find the end point of the current batch 

        cur_batch_size = batch_size #len(real) # Enas, changed
        # extract real images
        x1 = np.array([i for i in range(batch_size)]) # Generate numbers in the range of the batch size
        idx = rng.choice(x1, size=batch_size, replace=False) # note that this is the same as the batchsize but is ok and consider it as suffling the current minibatch 
        idx = x1
        idx = idx + start_index # add the batch index so you get indicies inside the next minibatch 

        #print(iteration)
        #print('Descriminator indicies for training', idx)
        #idx = np.array([i for i in range(dataset_size_train_split)]) # xtract the data from 0-5 as training dataif we want to include all of them sequentially 
        '''
        imgs = X_train[idx] # represent real images 
        real = imgs.to(device)
        '''
        real = X_train[idx] # represent real images 
        real = torch.from_numpy(real).float()
        real = real.to(device)

        '''
        z = z_norm[idx] # Enas, fake images  Changed 
        z = z.to(device)
        '''
        '''
        z = z_norm[idx] # Enas, fake images  Changed 
        z = torch.from_numpy(z).float()
        z = z.to(device)
        #print(z.shape)
        '''
        # Enas, this block has been moved so I can train on same images take for descriminator 
         # Enas, generate random images from the batch -> may be you can try next to sample from the dataset

        x1 = np.array([i for i in range(batch_size)]) # Generate numbers in the range of the batch size
        idx = rng.choice(x1, size=batch_size, replace=False) # note that this is the same as the batchsize but is ok and consider it as suffling the current minibatch 
        idx = x1
        idx = idx + start_index # add the batch index so you get indicies inside the next minibatch 

        # Enas. changed so that real and corresponding fake are paired. added so I an take the same idx for real and fake 

        real = X_train[idx] # represent real images 
        real = torch.from_numpy(real).float()
        real = real.to(device)


        z = z_norm[idx] # Enas, fake images  Changed
        z = torch.from_numpy(z).float()
        z = z.to(device)
            #fake = gen(itr_z) # Enas: changed
            #print(z.shape)
        
        #z = real # for training on real samples 
        
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            
            ### Update critic ###
            crit_opt.zero_grad()
            #fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            #itr_z = next(iter(z))

           
            fake = gen(z) # Enas: changed
            #print(real.shape)
            #print(fake.shape)
            # print(fake.shape)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)
            #print(crit_real_pred)
            #print(crit_real_pred.shape)
            
            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        # fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)

        # extract fake images for training the generator
        # Enas, below are comented so I can pair the fake and real images.
        
        '''
        x1 = np.array([i for i in range(batch_size)]) # Generate numbers in the range of the batch size
        idx = rng.choice(x1, size=batch_size, replace=False) # note that this is the same as the batchsize but is ok and consider it as suffling the current minibatch 
        idx = idx + start_index # add the batch index so you get indicies inside the next minibatch 

        z = z_norm[idx] # Enas, fake images  Changed
        z = torch.from_numpy(z).float()
        z = z.to(device)
        
        '''

        fake_2 = gen(z) # Enas: need to feed the features to the generator
        
        crit_fake_pred = crit(fake_2)
        
        
        
        fake_pred_on_real = gen(real)
        
        gen_loss = get_gen_loss(crit_fake_pred, fake_2, real, fake_pred_on_real, epoch+1)
        gen_loss.backward()
        
        torch.cuda.empty_cache()


        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]
        #reconst_loss+=[rec_loss.item()]
        # compute the mse loss for the generated images

                
        fake = gen(z)


        

        #----------------------------------------------------------------------
        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            #show_tensor_images(fake) commented by me 
            #show_tensor_images(real) commented by me 
            
            plt.imshow(fake[0,0,:,:].detach().cpu().numpy())
            plt.show()

            plt.imshow(real[0,0,:,:].detach().cpu().numpy())
            plt.show()

            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1

    # -----------------------------------------------------------------------
    #       Store loss per epoch only
    #-------------------------------------------------------------------------
    
    generator_losses_epoch += [gen_loss.item()]
    #reconst_loss_epoch +=[rec_loss.item()]
    critic_losses_epoch+=[mean_iteration_critic_loss]
    
    #-------------------------------------------------------------------------
    # save stats after each epoch 
    #------------------------------------------------------------------------
    valid_set_sample = validation_set_sample[:]
    #valid_set_sample =  torch.from_numpy(valid_set_sample).float() ,,,,,
    #valid_set_sample = valid_set_sample.to(device)
    zzz = np.zeros((1,1,num_nodes,num_nodes))
    gen = gen.to('cpu')# send the gpu to the cpu for validation 
    
    for i in range(valid_dataset_size_all): # Loop for all instances in the dataset; valid_dataset_size_all
      
      zzz[0,:,:,:] = z_norm_valid[i] # Enas, fake images  Changed
      zzz_valid = torch.from_numpy(zzz).float()
      #zzz_valid = zzz_valid.to(device)
      
      '''
      In case of identity 
      zzz[0,:,:,:] = validation_set_sample[i] # Enas, fake images  Changed
      zzz_valid = torch.from_numpy(zzz).float()
      zzz_valid = zzz_valid.to(device)
      '''
      
      gen_imgs = gen(zzz_valid) # test prediction on validation dataset. 

      array2 = gen_imgs[0,0,:,:].detach().cpu().numpy()#gen_imgs[0,0,:,:].detach().cpu().numpy()
      array1 = valid_set_sample[i,0,:,:] #valid_set_sample[i,0,:,:].detach().numpy()#valid_set_sample[i,0,:,:].detach().cpu().numpy()

      difference_array = np. subtract(array2, array1)
      squared_array = np. square(difference_array)
      #mse = squared_array. mean()
      #mse_each[epoch,i] = squared_array.mean() # save mse for each image
      mse_each_long_term_training[epoch] = squared_array.mean()
      
      #mse_all[epoch] = mse_each[epoch,:].mean() # store the mean of all values of images in the validation set
      mse_long_term_training[epoch] = mse_each_long_term_training[epoch,:].mean()
       
    # cut results go here .....
    
    if (epoch>0):# if this is not the first epoch
    
        if (mse_long_term_training[epoch] < mse_best):
            # save the generator model 
            torch.save({
                'epoch':epoch,
                'model_state_dict':gen.state_dict(),
                'optimizer_state_dict':gen_opt.state_dict(),
                'Loss':generator_losses,
            },PATH_G_Best)    
  
            # save the descriminator model 
            torch.save({
                'epoch':epoch,
                'model_state_dict':crit.state_dict(),
                'optimizer_state_dict':crit_opt.state_dict(),
                'Loss':critic_losses,
            },PATH_D_Best)
            
            mse_best = mse_long_term_training[epoch]
            
    else:# if this the first epoch
      # save the generator model
      torch.save({
          'epoch':epoch,
          'model_state_dict':gen.state_dict(),
          'optimizer_state_dict':gen_opt.state_dict(),
          'Loss':generator_losses,
      },PATH_G_Best)    

      # save the descriminator model 
      torch.save({
          'epoch':epoch,
          'model_state_dict':crit.state_dict(),
          'optimizer_state_dict':crit_opt.state_dict(),
          'Loss':critic_losses,
      },PATH_D_Best)
      mse_best = mse_long_term_training[epoch]
      
      
    '''
    # plot the results
    dr = np.mean(mse_each[0:38,0:31], axis=1)
    plt.plot(dr)
    plt.xlabel('epoch')
    plt.ylabel('mse all graphs')
    '''
    '''
    plt.plot(mse_each[0:38,0:31])
    plt.xlabel('epoch')
    plt.ylabel('mse per graph')
    '''
   
    gen = gen.to('cuda')
    # save the generator model 
    torch.save({
        'epoch':epoch,
        'model_state_dict':gen.state_dict(),
        'optimizer_state_dict':gen_opt.state_dict(),
        'Loss':generator_losses,
    },PATH_G)    

    # save the descriminator model 
    torch.save({
        'epoch':epoch,
        'model_state_dict':crit.state_dict(),
        'optimizer_state_dict':crit_opt.state_dict(),
        'Loss':critic_losses,
    },PATH_D)




'''
plt.plot(
     
     torch.Tensor(reconst_loss),
     label="Generator Loss"
 )

plt.plot(
    
    torch.Tensor(generator_losses),
    label="Generator Loss"
)
'''
end_time = start_time - time.time()


#=========================this is the part of the validation and it is sitored in the other files==============




PATH_G_Best = "WGAN_model_Gen_best.pt" # specify the location to save the model parameters.
PATH_D_Best = "WGAN_model_Desc_best.pt"



device = 'cpu'
checkpoint_G = torch.load(PATH_G_Best, map_location=device)
gen.load_state_dict(checkpoint_G['model_state_dict'],strict=False)
gen_opt.load_state_dict(checkpoint_G['optimizer_state_dict'])
epoch = checkpoint_G['epoch']
generator_losses = checkpoint_G['Loss']

# Second: Load for descriminator
checkpoint_D = torch.load(PATH_D_Best, map_location=device)
crit.load_state_dict(checkpoint_D['model_state_dict'],strict=False)
crit_opt.load_state_dict(checkpoint_D['optimizer_state_dict'])
epoch = checkpoint_D['epoch']
critic_losses = checkpoint_D['Loss']

import time
import argparse
import pprint as pp
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from Gurobi_tsp_reader import GurobiTSPReader

#dataset_size_test =1 #1000#600#8501
image_height,image_width = 20,20
num_nodes = 20
num_nodes = image_height 


rep_factor = 1
num_nodes = 20
valid_dataset_size_all = 100#600#850
mse_valid = np.zeros((1,valid_dataset_size_all))# store the mse over individual images
mae_valid = np.zeros((1,valid_dataset_size_all))# store the mse over individual images
xx_valid = np.zeros((valid_dataset_size_all * rep_factor,1,num_nodes,num_nodes)) # changed to mtch pytorch 
org_cost = np.zeros((valid_dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # changed to match pytorch 
z_norm_valid = np.zeros((valid_dataset_size_all* rep_factor,1,num_nodes,num_nodes)) # changed to match pytorch 
optimal_cost = np.zeros((valid_dataset_size_all * rep_factor,1,num_nodes,num_nodes))
optimal_tour_nodes =  np.zeros((valid_dataset_size_all * rep_factor,num_nodes))
optimal_tour_len = np.zeros((valid_dataset_size_all * rep_factor,1))

counter = 0

 
#train_filepath = r"C:\Users\eo2fg\tsp10_test_Gurobi.txt" 
num_neighbors= -1
batch_size= 1 # Warning =;;;;;; next you may not use it s one .
#train_filepath = r"C:\Users\eo2fg\tsp10_test_Gurobi_30_45.txt" 
test_filepath = f"mmwave{num_nodes}_test_Gurobi.txt"#r"C:\Users\eo2fg\20_nodes_mmwave_0_105_test_v03.txt"
test_filepath = f"mmwave{num_nodes}_Gurobi_0_105_Throughput_obj_v02_zero_cnr_2_temp_.txt"

dataset = GurobiTSPReader(num_nodes, num_neighbors, batch_size, test_filepath)
print("Number of batches of size {}: {}".format(batch_size, dataset.max_iter))
batch = next(iter(dataset))  # Generate a batch of TSPs
start_time = time.time()

i = iter(dataset)

index = 0    

for itr_num in range( np.int32(valid_dataset_size_all)): # np.int32(valid_dataset_size_all)in case of repetition dataset_size_all/rep_factor
    
    next_batch = next(i)
    Cost = np.multiply(next_batch.edges_values,next_batch.edges_target)
    
    opt_adj_2d = next_batch.edges_target
    
    '''
    # Get the optimal adjacency matricies
    # In case you want upper triangular matrix 
    dff_masked = np.triu(opt_adj_2d[0,:,:], k =0) # create upper triangulau matrix of dff
    dff_mat_diag = np.diag(np.diag(opt_adj_2d[0,:,:])) # create diagonal matrix with diagonal elements of the dff matrix 
    diff_mat = dff_masked - dff_mat_diag # mask the digonal elements also    
    xx_valid[itr_num] = diff_mat
    '''
    
    #for j in range(rep_factor): # reat the data 4 times
      #xx[counter+j] = diff_mat
    
    # In case you want the full matrix 
    xx_valid[index] = opt_adj_2d
    '''
    # Get the original cost matrix
    orig_cost = next_batch.edges_values
    dff_masked = np.triu(orig_cost[0,:,:], k =0) # create upper triangulau matrix of dff
    dff_mat_diag = np.diag(np.diag(orig_cost[0,:,:])) # create diagonal matrix with diagonal elements of the dff matrix 
    diff_mat = dff_masked - dff_mat_diag # mask the digonal elements also 
    z_norm_valid[itr_num] = diff_mat
    '''
    #for j in range(rep_factor):
      #z_norm[counter+j] = diff_mat
      
    #counter = counter+rep_factor  
    
    
    # In case that you want the ful matrix 
    z_norm_valid[index] = next_batch.edges_values
    
    
    # Get the optimal tour
    optimal_tour_len[index]=next_batch.tour_len
    optimal_tour_nodes[index] = next_batch.tour_nodes
    index = index + 1

# *********************************Search code ********************************
# This is the code of search.  
    
from math import log
from numpy import array
from numpy import argmax
import numpy as np
import pandas as pd
from logging import root
import numpy

  
#=========================this is the part of the validation and it is sitored in the other files=====================================================
from utils.graph_utils import *
from utils.model_utils import *
beam_size = 1

'''
if torch.cuda.is_available():
    #print("CUDA available, using GPU ID {}".format(config.gpu_id))
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)
'''

print("CUDA not available")
dtypeFloat = torch.FloatTensor
dtypeLong = torch.LongTensor
torch.manual_seed(1)
device = 'cpu'
batch_size = 1
    
# Step 3: Compute the generator output 
num_nodes = 20


shift_index = 0
zzz = np.zeros((1,1,num_nodes,num_nodes))
xxx = np.zeros((1,1,num_nodes,num_nodes))
gap_in_tour_len = np.zeros((1,valid_dataset_size_all))
gen_tour_len = np.zeros((valid_dataset_size_all,1))
opt_tour_extracted = np.zeros((valid_dataset_size_all,1))
mse_ones_zeros =   np.zeros((1,valid_dataset_size_all))# 
squared_array_all =   np.zeros((1,valid_dataset_size_all))#
gen_tour_len_our_search = np.zeros((valid_dataset_size_all,1))     
path_len_gap_our_search =  np.zeros((valid_dataset_size_all,1))
optimality_gap = np.zeros((valid_dataset_size_all,1))
optimality_gap_our_search=np.zeros((valid_dataset_size_all,1))

gen_tour_len_beam_search = np.zeros((valid_dataset_size_all,1))     
path_len_gap_beam_search =  np.zeros((valid_dataset_size_all,1))
generated_final_tour = np.zeros((valid_dataset_size_all,num_nodes))
unique_nodes = np.zeros((valid_dataset_size_all,1))
tour_is_valid = np.zeros((valid_dataset_size_all,1))
start_time = time.time()
beam_size = 1280
fig, axs = plt.subplots(5,3,figsize=(10, 10),sharex='col', sharey='row',)

batch_size = 1
optimal_rate = np.zeros((valid_dataset_size_all,1)) 

rate_values = np.zeros((num_nodes,1))
nn_optimal_rate = np.zeros((valid_dataset_size_all,1)) 
Gurobi_optimal_rate = np.zeros((valid_dataset_size_all,1))

nn_final_tour = np.zeros((valid_dataset_size_all,num_nodes))
nn_optimality_gap = np.zeros((valid_dataset_size_all,1))

nn_tour_len_all = np.zeros((valid_dataset_size_all,1))
greedy_tour_len_all = np.zeros((valid_dataset_size_all,1))

greedy_optimal_rate = np.zeros((valid_dataset_size_all,1)) 
greedy_final_tour = np.zeros((valid_dataset_size_all,num_nodes))
greedy_optimality_gap = np.zeros((valid_dataset_size_all,1))
import tspsolve
from tsp_solver.greedy import solve_tsp
bandwidth = 500*1e6 
power = 4

beam_search_edge = np.zeros((num_nodes,num_nodes))

for i in range(valid_dataset_size_all):#v alid_dataset_size_all valid_dataset_size_all

  
    beam_search_edge = np.zeros((num_nodes,num_nodes))
    zzz[0,:,:,:] = z_norm_valid[i+shift_index] # Enas, fake images  Changed
    zzz_valid = torch.from_numpy(zzz).float()
    zzz_valid = zzz_valid.to(device)
    
    xxx[0,:,:,:] = xx_valid[i+shift_index]
    xxx_valid = torch.from_numpy(xxx).float()
    xxx_vlid = xxx_valid.to(device)
   
    fake_valid = gen(zzz_valid) 
    
    
    # Compute the mse loss over the generated image
    #array2 = fake_valid[0,0,:,:].detach().cpu().numpy()
   
    #print(mse_valid)
    #print(squared_array.mean())
    a = xxx_valid[0,:,:].detach().cpu().numpy()
    
    '''
    # display the generated and the optimal one
    plt.imshow(fake_valid[0,0,:,:].detach().cpu().numpy()) # cost of the original graph 
    plt.show()
    
    a = xxx_valid[0,:,:].detach().cpu().numpy()
    
    plt.imshow(a[0,:,:]) # the adjacency matrix of the optimal route I choose 8 bcause it is the last one in the validation set 
    plt.show()
    '''
    
    # I want to compute the optimal tour length and compre it with generated one.

    #print(fake_valid[0,0,:,:].detach().cpu().numpy())
    image_to_handle = fake_valid[0,0,:,:].detach().cpu().numpy()
    image_to_handle_tensor = torch.from_numpy(image_to_handle) # convert ndarray to tensor
    y = torch.zeros((num_nodes,num_nodes))
    image_to_handle_thresh = torch.where(image_to_handle_tensor >0, image_to_handle_tensor, y)#>0.1
    
    y = torch.ones((num_nodes,num_nodes))
    image_to_handle_thresh = torch.where(image_to_handle_thresh >1e-03, y, image_to_handle_thresh) # warning; commented to add search method 
    
    # -------------------------------------------------------------------------
    #                       # show the original and the generated output
    #--------------------------------------------------------------------------
    '''
    axs[i,0].imshow(fake_valid[0,0,:,:].detach().cpu().numpy())
    axs[i,1].imshow(image_to_handle_thresh)
    axs[i,2].imshow(a[0,:,:]) 
    '''
    # --------------------------------------------------------------------------------------
    # Attention: this code has been added to compute the optimal tour using beam search 
    #---------------------------------------------------------------------------------------
    
    y_preds = fake_valid[:,:,:,:]#.detach().cpu()#.numpy()
    #cost_values = torch.from_numpy(zzz_vaid).float()
    
    cost_values = z_norm_valid[i+shift_index,0,:,:]
    
    
    #bs_nodes = beamsearch_tour_nodes(
                        #y_preds, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')   

    bs_nodes,tour_is_valid[i] = beamsearch_tour_nodes_shortest(y_preds, cost_values, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='logits', random_start=False)## 18.578189682059765 = beam_size = 1, ot the best 18.578189682059765, beam_1280=8.088487828457257 with time = 2.463889, number of invalid tours=232  

    # now, compute the length of the generated tour using the beam search 
    cur_node_to_investigate = i # the graph to which we compute the generated output 
    #orig_graph_cost = z_norm_valid[cur_node_to_investigate+shift_index,0,:,:] # In the case of full matrix
    
    orig_graph_cost = cost_values # only in the case of upper triangle matrix 
    
   
    
    # Now, it is required to find the tour length of the searched generated 
    # The optimal tour length is stored in optimal_tour_len    
    gen_tour_len = 0
    index_1 = 0
    index_2 = 0
    
    f = bs_nodes.tolist()
    
    #f = bs_nodes[0].tolist()
    final_tour = f[0]
    
    generated_final_tour[i] = final_tour
    
    for k in range(num_nodes-1): # range number of (nodes - 1)
        
        index_1 = int(final_tour[k])
        index_2 = int(final_tour[k+1])
        gen_tour_len = gen_tour_len + orig_graph_cost[index_1][index_2]  
        cnr = 1/cost_values[index_1][index_2]
        rate_values[k] = bandwidth*np.log2(1+power*cnr)
        beam_search_edge[index_1][index_2] = 1
        beam_search_edge[index_2][index_1] = 1
    
    index_1 = int(final_tour[num_nodes - 1])
    start_node_of_tour = 0 # this is the starting node in the tour 
    index_2 = start_node_of_tour    
    cnr = 1/cost_values[index_1][index_2] 
    rate_values[k+1] = bandwidth*np.log2(1+power*cnr)
    optimal_rate[i] = np.min(rate_values)
    beam_search_edge[index_1][index_2] = 1
    beam_search_edge[index_2][index_1] = 1
    
    gen_tour_len_beam_search[i] = gen_tour_len + orig_graph_cost[index_1][index_2]     

    # find the gap between the searched path length and the optimal one
    path_len_gap_beam_search[i] =  optimal_tour_len[cur_node_to_investigate+shift_index] - gen_tour_len_beam_search[i] # either this or you can add the absolute value 
    optimality_gap[i] =  1 - (optimal_tour_len[cur_node_to_investigate+shift_index]/gen_tour_len_beam_search[i])#(gen_tour_len_beam_search[i]/optimal_tour_len[cur_node_to_investigate+shift_index])-1
    
    optimality_gap[i] = optimality_gap[i]*100
    x_unique = np.unique(final_tour)
    unique_nodes[i] = x_unique.size
    end_time = start_time - time.time()
    
    #==============================================================================
    # This is the computation of the optimal tour length based on the output of Gurobi 
    #==============================================================================
    
    index_1 = 0
    index_2 = 0
    
    final_tour = optimal_tour_nodes[i]
    for k in range(num_nodes-1): # range number of (nodes - 1)       
        index_1 = int(final_tour[k])
        index_2 = int(final_tour[k+1])  
        cnr = 1/cost_values[index_1][index_2]
        rate_values[k] = bandwidth*np.log2(1+power*cnr)
           
    index_1 = int(final_tour[num_nodes - 1])
    start_node_of_tour = 0 # this is the starting node in the tour 
    index_2 = start_node_of_tour    
    cnr = 1/cost_values[index_1][index_2] 
    rate_values[k+1] = bandwidth*np.log2(1+power*cnr)
    Gurobi_optimal_rate[i] = np.min(rate_values)
 
    # =========================================================================
    #This is the implemntation of the nearest neighbout solution 
    #==========================================================================
    
    nn_tour_len = 0
    index_1 = 0
    index_2 = 0
    
    path = tspsolve.nearest_neighbor(cost_values)
    path = tspsolve.two_opt(cost_values, path, verbose=True)
    
    #f = bs_nodes[0].tolist()
    final_tour = path 
    
    nn_final_tour[i] = final_tour
    
    for k in range(num_nodes-1): # range number of (nodes - 1)
        
        index_1 = int(final_tour[k])
        index_2 = int(final_tour[k+1])
        nn_tour_len = nn_tour_len + orig_graph_cost[index_1][index_2]  
        coeff_val = 1/cost_values[index_1][index_2]
        cnr = 1/cost_values[index_1][index_2] 
        rate_values[k] = bandwidth*np.log2(1+power*cnr)
        
        
    
    index_1 = int(final_tour[num_nodes - 1])
    start_node_of_tour = final_tour[0]# 0 # this is the starting node in the tour 
    index_2 = start_node_of_tour    
    cnr = 1/cost_values[index_1][index_2]
    rate_values[k+1] = bandwidth*np.log2(1+power*cnr)    
    nn_optimal_rate[i] = np.min(rate_values)
    
  
    nn_tour_len_all[i] = nn_tour_len + orig_graph_cost[index_1][index_2]     

    # find the gap between the searched path length and the optimal one
   
    nn_optimality_gap[i] =  (nn_tour_len_all[i]/optimal_tour_len[cur_node_to_investigate+shift_index])-1
    nn_optimality_gap[i] = nn_optimality_gap[i]*100
   
    #=====================================================
    # This is the greedy solution 
    #====================================================
    
    greedy_tour_len = 0
    index_1 = 0
    index_2 = 0
  
    path = solve_tsp(cost_values,endpoints = (0,0))
    
    #f = bs_nodes[0].tolist()
    final_tour = path 
    
    greedy_final_tour[i] = final_tour[0:num_nodes]
    
    for k in range(num_nodes): # I have done this because its 0 to 0 here.
        
        index_1 = int(final_tour[k])
        index_2 = int(final_tour[k+1])
        greedy_tour_len = greedy_tour_len + cost_values[index_1][index_2]  
        coeff_val = 1/cost_values[index_1][index_2]
        cnr = 1/cost_values[index_1][index_2]
        rate_values[k] = bandwidth*np.log2(1+power*cnr)    
        
    
    greedy_optimal_rate[i] = np.min(rate_values)
    
    greedy_tour_len_all[i] = greedy_tour_len #+ orig_graph_cost[index_1][index_2]     

    # find the gap between the searched path length and the optimal one
   
    greedy_optimality_gap[i] =  (greedy_tour_len_all[i]/optimal_tour_len[cur_node_to_investigate+shift_index])-1
    greedy_optimality_gap[i] = greedy_optimality_gap[i]*100
    
