import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import argparse
import pprint as pp
import os
from numpy.random import default_rng
from utils.graph_utils import *
from Gurobi_tsp_reader import GurobiTSPReader
from logging import root
from utils.model_utils import *
import matplotlib.pyplot as plt
import networkx as nx


rng = default_rng()
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



padding= (7,7,7,7)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim//2 , out_channels = in_dim//2 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B * N * C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        energy =  torch.bmm(proj_query, proj_key) # batch matrix-matrix product
        attention = self.softmax(energy) # B * N * N
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B * C * N
        out = torch.bmm(proj_value, attention.permute(0,2,1)) # batch matrix-matrix product
        out = out.view(m_batchsize,C,width,height) # B * C * W * H
        out = self.gamma*out + x
        return out 

class Generator(nn.Module):

    def __init__(self): 
        
        im_chan=1
        hidden_dim=64
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConstantPad2d(padding, value=0),
            nn.Conv2d(im_chan,hidden_dim * 4, kernel_size=26,stride=2),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            Self_Attn(hidden_dim*2),
            self.make_gen_block(hidden_dim, hidden_dim, kernel_size=4, final_layer=False),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=5,stride=1, final_layer=False),
            nn.Conv2d(im_chan,im_chan,kernel_size = 2, stride=2), 
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



class Critic(nn.Module):

    
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            nn.ConstantPad2d(padding, value=0),
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
    
    

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def get_gradient(crit, real, fake, epsilon):

    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty

def get_gen_loss(crit_fake_pred, fake, real,g_on_real_pred,epoch_num):
    
    lambda_recon = 200
    recon_criterion =    nn.L1Loss() 
    gen_rec_loss = recon_criterion(real, fake)
    
    recon_criterion_iden =  nn.L1Loss() 
    identity_loss = recon_criterion_iden(real, g_on_real_pred)
    gen_loss = -1. * torch.mean(crit_fake_pred) + (lambda_recon) * gen_rec_loss+ identity_loss
    
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp 
    return crit_loss

# -------------------Dataset Preparation ------------------------------------

def load_dataset(train_filepath,val_filepath,num_nodes = 50,train_dataset_size = 1000000, valid_dataset_size_all = 10000):
    
   
    num_neighbors= -1
    batch_size= 1
    # Read training dataset 
    train_dataset = GurobiTSPReader(num_nodes, num_neighbors, batch_size, train_filepath)
    xx = np.zeros((train_dataset_size ,1,num_nodes,num_nodes)) # 
    z_norm = np.zeros((train_dataset_size,1,num_nodes,num_nodes)) #
    optimal_tour_len_train = np.zeros((train_dataset_size,1))
    optimal_tour_nodes_train = np.zeros((train_dataset_size,num_nodes))
    i = iter(train_dataset)
    for itr_num in range(np.int32(train_dataset_size)):
        
        next_batch = next(i)
        xx[itr_num] = next_batch.edges_target
        z_norm[itr_num] = next_batch.edges_values
        # Get the optimal tour
        optimal_tour_len_train[itr_num]=next_batch.tour_len # Get the optimal tour
        optimal_tour_nodes_train[itr_num] = next_batch.tour_nodes  # Get the tour nodes 
    
    # plot matrix for testing validity 
    plt.imshow(xx[0,0,:,:])
    plt.show()
    
    plt.imshow(z_norm[0,0,:,:])
    plt.show()
    
    # Read validation dataset
    val_dataset = GurobiTSPReader(num_nodes, num_neighbors, batch_size, val_filepath)
    validation_set_sample = np.zeros((valid_dataset_size_all,1,num_nodes,num_nodes)) # 
    z_norm_valid = np.zeros((valid_dataset_size_all,1,num_nodes,num_nodes)) # 
    optimal_tour_len_val = np.zeros((valid_dataset_size_all,1))
    optimal_tour_nodes_val = np.zeros((valid_dataset_size_all,num_nodes))
    
    i = iter(val_dataset)
    for valid_itr_num in range(np.int32(valid_dataset_size_all)):#
        
        
        next_batch = next(i)
        validation_set_sample[valid_itr_num] = next_batch.edges_target
        z_norm_valid[valid_itr_num] = next_batch.edges_values
        
        # Get the optimal tour
        optimal_tour_len_val[valid_itr_num]=next_batch.tour_len # Get the optimal tour
        optimal_tour_nodes_val[valid_itr_num] = next_batch.tour_nodes  # Get the tour nodes 
        
    return xx,z_norm,validation_set_sample,z_norm_valid

#===========================================================================================================================
#                                                   Training function                                                      #
#===========================================================================================================================

def train_model(gen,crit,num_nodes,batch_size,train_filepath,val_filepath,train_dataset_size,valid_dataset_size,n_epochs,pretrained=False,load_best=True):
  
    '''
    train_filepath: is the file containig training dataset
    val_filepath: is the file containig validation dataset
    n_epochs: Number of epochs to train for 
    pretrained: if pretrained model to be used
    load_best: if oretrained: either best model trained so far or the recent one.
    '''
    
    X_train, z_norm,validation_set_sample,z_norm_valid  = load_dataset(train_filepath,val_filepath,num_nodes,train_dataset_size, valid_dataset_size)
    scale_factor = 1
    #train_dataset_size = X_train.shape[0] #train_dataset_size: Size of the training data
    #valid_dataset_size = validation_set_sample.shape[0]
    batch_index = [n for n in range(0, train_dataset_size, batch_size)]
    mse_all = np.zeros((scale_factor*n_epochs, 1))
    generator_losses = []
    critic_losses = []
    end_index =  0 # before strting the training it should start from 0
    cur_step = 0
    scale_factor = 1 # just if the epochs needs to be increased
    generator_losses_epoch=[]
    critic_losses_epoch = []
    step = 0
    tick_width = 50
    #mse_val = np.zeros((scale_factor*n_epochs, 1))
    mse_val = []
    mse_per_inst_val = np.zeros((scale_factor*n_epochs, valid_dataset_size))
    
    
    PATH_G = "WGAN_model_Gen.pt" 
    PATH_D = "WGAN_model_Desc.pt"
    PATH_G_Best = "WGAN_model_Gen_best.pt"
    PATH_D_Best = "WGAN_model_Desc_best.pt"

    # if pretrained models are to be used
    if(pretrained and load_best):
        
        # load the model
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
       
    elif(pretrained and ~load_best): 
        
        checkpoint_G = torch.load(PATH_G, map_location=device)
        gen.load_state_dict(checkpoint_G['model_state_dict'],strict=False)
        gen_opt.load_state_dict(checkpoint_G['optimizer_state_dict'])
        epoch = checkpoint_G['epoch']
        generator_losses = checkpoint_G['Loss']

        # Second: Load for descriminator
        checkpoint_D = torch.load(PATH_D, map_location=device)
        crit.load_state_dict(checkpoint_D['model_state_dict'],strict=False)
        crit_opt.load_state_dict(checkpoint_D['optimizer_state_dict'])
        epoch = checkpoint_D['epoch']
        critic_losses = checkpoint_D['Loss']
        
    with open('Model_results_summary.txt',"a" , encoding="utf-8") as f:
       
       f.write('Training Results statistics')
       f.write("Number of nodes = " + str(num_nodes))
       f.write("Number of training samples = " + str(train_dataset_size))
       f.write("Number of validation samples = " + str(valid_dataset_size))
       
    print('Training over epochs started ....')
    start = time.time()
    mse_best = 100 # best mse encountered so far in case to continue training
    # training loop over all dataset
    for epoch in range(scale_factor*n_epochs):
        #torch.cuda.empty_cache()
        end_index = 0
        start_index = end_index # find the starting point of the current batch       
        # training loop
        for iteration in range(len(batch_index)-1): #

           start_index = end_index # find the starting point of the current batch
           end_index = start_index + batch_size # find the end point of the current batch 
           cur_batch_size = batch_size 
           # extract real adjacency matrix
           x1 = np.array([i for i in range(batch_size)]) # Generate numbers in the range of the batch size
           idx = x1 + start_index 

           real = X_train[idx] 
           real = torch.from_numpy(real).float()
           real = real.to(device)

           z = z_norm[idx] #
           z = torch.from_numpy(z).float()
           z = z.to(device)
           
           mean_iteration_critic_loss = 0
           for _ in range(crit_repeats):
    
               crit_opt.zero_grad()
               fake = gen(z) # Enas: changed
               crit_fake_pred = crit(fake.detach())
               crit_real_pred = crit(real)
               epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
               gradient = get_gradient(crit, real, fake.detach(), epsilon)
               gp = gradient_penalty(gradient)
               crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
               # mean critic loss per current batch
               mean_iteration_critic_loss += crit_loss.item() / crit_repeats
               # Update the critic gradients
               crit_loss.backward(retain_graph=True)
               # Update critic optimizer
               crit_opt.step()
           critic_losses += [mean_iteration_critic_loss]
           gen_opt.zero_grad() # update the generator

           fake_2 = gen(z) # Enas: need to feed the features to the generator
           crit_fake_pred = crit(fake_2)
           fake_pred_on_real = gen(real)
           gen_loss = get_gen_loss(crit_fake_pred, fake_2, real, fake_pred_on_real, epoch+1)
           gen_loss.backward()
           
           torch.cuda.empty_cache()
           gen_opt.step() # Update the weights
           generator_losses += [gen_loss.item()] # mean loss per iteration 
           
           # prediction for visulaization 
           fake = gen(z)

           #------------------------need to update this----------------------------
           ### Visualization code ###
           # The code display the results every num_examples by dividing it into chunks of step_size and average over the whole block
           if step % display_step == 0 and step > 0:
               
               
               gen_mean = sum(generator_losses[-display_step:]) / display_step # This is the moving average every dispay step 
               crit_mean = sum(critic_losses[-display_step:]) / display_step
               print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
              
               plt.imshow(fake[0,0,:,:].detach().cpu().numpy())
               plt.show()

               plt.imshow(real[0,0,:,:].detach().cpu().numpy())
               plt.show()

               
               plt.plot(
                   range(num_data_chuncks // tick_width), 
                   torch.Tensor(critic_losses[:num_data_chuncks]).view(-1, tick_width).mean(1),
                   label="Discriminator Loss"
               )
               num_data_chuncks = (len(generator_losses) // tick_width) * tick_width
               plt.plot(
                   range(num_data_chuncks // tick_width), 
                   torch.Tensor(generator_losses[:num_data_chuncks]).view(-1, tick_width).mean(1),
                   label="Generator Loss"
               )
              
               plt.legend()
               plt.show()

           step += 1

       # -----------------------------------------------------------------------
       #       Store loss per epoch only
       #-------------------------------------------------------------------------
       
        generator_losses_epoch += [gen_loss.item()]
        critic_losses_epoch+=[mean_iteration_critic_loss]
       
       #-------------------------------------------------------------------------
       # save stats after each epoch 
       #------------------------------------------------------------------------
        valid_set_sample = validation_set_sample[:]
        zzz = np.zeros((1,1,num_nodes,num_nodes))
        gen = gen.to('cpu')
        print('start validation')
        for i in range(valid_dataset_size): # Loop for all instances in the dataset; valid_dataset_size_all
         
          zzz[0,:,:,:] = z_norm_valid[i] 
          zzz_valid = torch.from_numpy(zzz).float()
        
          gen_imgs = gen(zzz_valid) # predict

          array2 = gen_imgs[0,0,:,:].detach().cpu().numpy()
          array1 = valid_set_sample[i,0,:,:] 

          difference_array = np. subtract(array2, array1)
          squared_array = np. square(difference_array)
          mse_per_inst_val[epoch] = squared_array.mean()
          
       
        #mse_val[epoch] = mse_per_inst_val[epoch,:].mean()
        mse_val.append(mse_per_inst_val[epoch,:].mean())
        with open('Model_results_summary.txt',"a" , encoding="utf-8") as f:
            f.write("Validation mean square error = "+ str(mse_val[epoch])) # store the validation mean square error 
        print(f'Epoch {epoch}/{n_epochs}:----training mse = , validation mse = {mse_val[epoch]}, Generator loss: {np.array(generator_losses_epoch).mean()}, critic loss: {np.array(critic_losses_epoch).mean()}')
        
        if (epoch>0):# if this is not the first epoch
        
            if (mse_val[epoch] < mse_best):
                mse_best = mse_val[epoch]
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
         mse_best = mse_val[epoch]
    
      
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
    
    end_time = time.time() - start
    with open('Model_results_summary.txt',"a" , encoding="utf-8") as f:
        f.write("best mse found = " + str(mse_best))
        f.write("training time = " + str(end_time))
    
       
       
    return gen, mse_val

#==============================================================================
#                       Testing Function 
#==============================================================================

def model_testing(gen,num_nodes, testing_datset_size, beam_size = 1280,test_filepath = None,load_best=True):
    
    device = 'cpu'
    gen = gen.to('cpu')
    num_neighbors= -1
    batch_size= 1
    org_cost = np.zeros((testing_datset_size,1,num_nodes,num_nodes)) # changed to match pytorch 
    z_norm_test = np.zeros((testing_datset_size,1,num_nodes,num_nodes)) # changed to match pytorch 
    optimal_cost = np.zeros((testing_datset_size,1,num_nodes,num_nodes))
    optimal_tour_nodes =  np.zeros((testing_datset_size,num_nodes))
    optimal_tour_len = np.zeros((testing_datset_size,1))
    xx_test = np.zeros((testing_datset_size ,1,num_nodes,num_nodes))
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)
    batch_size = 1
    beam_search_edge = np.zeros((num_nodes,num_nodes))
    zzz = np.zeros((1,1,num_nodes,num_nodes))
    xxx = np.zeros((1,1,num_nodes,num_nodes))
    tour_is_valid = np.zeros((testing_datset_size,1))
    gen_tour_len = np.zeros((testing_datset_size,1))
    optimality_gap = np.zeros((testing_datset_size,1))
    optimality_gap_in_cost = np.zeros((testing_datset_size,1))
    optimality_gap_in_thr = np.zeros((testing_datset_size,1))
    gen_tour_len_beam_search = np.zeros((testing_datset_size,1))
    path_len_gap_beam_search =  np.zeros((testing_datset_size,1))
    generated_final_tour = np.zeros((testing_datset_size,num_nodes))
    optimal_rate = np.zeros((testing_datset_size,1)) 
    Gurobi_optimal_rate = np.zeros((testing_datset_size,1))
    rate_values = np.zeros((num_nodes,1))
    rate_values_gen = np.zeros((num_nodes,1))
    gen_tour_len = 0
    index_1 = 0
    index_2 = 0
    shift_index = 0
    bandwidth = 500*1e6 
    power = 4
    
    #fig, axs = plt.subplots(5,3,figsize=(10, 10),sharex='col', sharey='row',)

    #Load the trained model
    if(load_best == True):
        
        PATH_G_Best = "WGAN_model_Gen_best.pt" # path to weight 
        PATH_D_Best = "WGAN_model_Desc_best.pt"
        checkpoint_G = torch.load(PATH_G_Best, map_location=device)
        checkpoint_D = torch.load(PATH_D_Best, map_location=device)# Second: Load for descriminator
        
    elif(load_best == False):#Load the trained model
    
        PATH_G  = "WGAN_model_Gen.pt" # path to weight 
        PATH_D = "WGAN_model_Desc.pt"
        checkpoint_G = torch.load(PATH_G, map_location=device)
        checkpoint_D = torch.load(PATH_D, map_location=device)# Second: Load for descriminator
    
    gen.load_state_dict(checkpoint_G['model_state_dict'],strict=False)
    gen_opt.load_state_dict(checkpoint_G['optimizer_state_dict'])
    epoch = checkpoint_G['epoch']
    generator_losses = checkpoint_G['Loss']
    crit.load_state_dict(checkpoint_D['model_state_dict'],strict=False)
    crit_opt.load_state_dict(checkpoint_D['optimizer_state_dict'])
    epoch = checkpoint_D['epoch']
    critic_losses = checkpoint_D['Loss']
    
    
    # read the dataset
    if test_filepath == None:
        test_filepath = f"mmwave{num_nodes}_test_Gurobi.txt"
    
    dataset = GurobiTSPReader(num_nodes, num_neighbors, batch_size, test_filepath)
    i = iter(dataset)
    print("Number of batches of size {}: {}".format(batch_size, dataset.max_iter))
    
    # Extract testing data
    for itr_num in range( np.int32(testing_datset_size)):
        
        next_batch = next(i)
        xx_test[itr_num] = next_batch.edges_target # store the optimal solution 
        z_norm_test[itr_num] = next_batch.edges_values # store the cost matrix.
        optimal_tour_len[itr_num]=next_batch.tour_len # store the optimal tour length 
        optimal_tour_nodes[itr_num] = next_batch.tour_nodes # store the final optimal tour 
 
    print(f'Testing phase starts, test datset size = {testing_datset_size}, number of nodes = {num_nodes}')
    # Inference loop
    start = time.time()
    for i in range(testing_datset_size):#

  
        beam_search_edge = np.zeros((num_nodes,num_nodes))
        zzz[0,:,:,:] = z_norm_test[i+shift_index] 
        zzz_test = torch.from_numpy(zzz).float()
        zzz_test = zzz_test.to(device)
        xxx[0,:,:,:] = xx_test[i+shift_index]
        xxx_test = torch.from_numpy(xxx).float()
        xxx_test = xxx_test.to(device)       
        
        fake_test = gen(zzz_test) 
        y_preds = fake_test[:,:,:,:]#.
      
        cost_values = z_norm_test[i+shift_index,0,:,:]
        # Beam Search
        bs_nodes,tour_is_valid[i] = beamsearch_tour_nodes_shortest(y_preds, cost_values, beam_size, batch_size, num_nodes,
                                       dtypeFloat, dtypeLong, probs_type='logits', random_start=False)
        
        
        cur_node_to_investigate = i # the graph to which we compute the generated output 
        
        f = bs_nodes.tolist()
        final_tour = f[0]
        generated_final_tour[i] = final_tour
        
        for k in range(num_nodes-1): # range number of (nodes - 1)
            
            index_1 = int(final_tour[k])
            index_2 = int(final_tour[k+1])
            gen_tour_len = gen_tour_len + cost_values[index_1][index_2]  
            cnr = cost_values[index_1][index_2]
            rate_values_gen[k] = bandwidth*np.log2(1+power*cnr)
            beam_search_edge[index_1][index_2] = 1
            beam_search_edge[index_2][index_1] = 1
        
        index_1 = int(final_tour[num_nodes - 1])
        start_node_of_tour = 0 # this is the starting node in the tour 
        index_2 = start_node_of_tour    
        cnr = cost_values[index_1][index_2] 
        rate_values_gen[k+1] = bandwidth*np.log2(1+power*cnr)
        optimal_rate[i] = np.min(rate_values_gen)
        beam_search_edge[index_1][index_2] = 1
        beam_search_edge[index_2][index_1] = 1
        
        gen_tour_len_beam_search[i] = gen_tour_len + cost_values[index_1][index_2]     
    
        # find the gap between the searched path length and the optimal one
        optimality_gap_in_cost[i] =  np.absolute(1 - (optimal_tour_len[cur_node_to_investigate+shift_index]/gen_tour_len_beam_search[i])) # absolute value of the rate becasue we might have negative
        optimality_gap_in_cost[i] = optimality_gap[i]*100
       
        
        # compute the throughput based on optimal solution
        index_1 = 0
        index_2 = 0
        
        final_tour = optimal_tour_nodes[i]
        for k in range(num_nodes-1): # range number of (nodes - 1)       
            index_1 = int(final_tour[k])
            index_2 = int(final_tour[k+1])  
            cnr = cost_values[index_1][index_2]
            rate_values[k] = bandwidth*np.log2(1+power*cnr)
               
        index_1 = int(final_tour[num_nodes - 1])
        start_node_of_tour = 0 # this is the starting node in the tour 
        index_2 = start_node_of_tour    
        cnr = cost_values[index_1][index_2] 
        rate_values[k+1] = bandwidth*np.log2(1+power*cnr)
        Gurobi_optimal_rate[i] = np.min(rate_values)
        
        optimality_gap_in_thr[i] =  np.absolute(1 - (Gurobi_optimal_rate[i]/optimal_rate[i])) 
        optimality_gap_in_thr[i] = optimality_gap_in_thr[i]*100
       
    end_time =  time.time() - start
    #device = 'cuda'
    print(f'End of the testing phase; Optimality gap in cost values = {np.mean(optimality_gap_in_cost)}, Optimality gap in throughput = {np.mean(optimality_gap_in_thr)}')
    
    with open('Model_results_summary.txt',"a" , encoding="utf-8") as f:
       
       f.write('Test results statistics')
       f.write("Number of nodes = " + str(num_nodes))
       f.write("Number od testing samples = " + str(testing_datset_size))
       f.write("Beam size = " + str(beam_size))
       f.write("Testing time = " + str(end_time))
       f.write("Optimality gap in throughput = "+ str(np.mean(optimality_gap_in_thr)))
       f.write("Optimality gap in cost matrix values = "+ str((np.mean(optimality_gap_in_cost)))) # this might be different because of the differences in the cost matrix 
       f.write("Final netowrk topology = "+ str(final_tour)) # the final tour generated after beam search

#==============================================================================
#                   plot results
#==============================================================================

def plot_model_results(mse_val,status):
    
    # plot the mse of the validation over epochs 
    plt.plot(mse_val, linewidth = 2)
    plt.title('MSE over iterations during' + status, fontsize = 14)
    plt.xlabel('Epoch number',fontsize = 14)
    plt.ylabel('MSE',fontsize = 14)
    #xticks(np.arange(0, mse_val.shape[0], step=1))  # Set label locations.
    plt.xticks(np.arange(0, mse_val.shape[0], step=1))
    plt.show()
    
    # To perform the following code then someone needs to generate dataset and store coordinates and the optimal solution.
    '''
    idx = 0
    fig_all = plt.figure(figsize=(13,13))
    a = fig_all.add_subplot(331)
    plot_tsp(a, next_batch.nodes_coord[idx], next_batch.edges[idx], next_batch.edges_values[idx], next_batch.edges_target[idx])
    a.set(xlim=[0, 2.2], ylim=[0, 2.2], ylabel='Distance in Km', xlabel='(a)')
    
    b = fig_all.add_subplot(332)
    plot_tsp(b, next_batch.nodes_coord[idx], next_batch.edges[idx], next_batch.edges_values[idx], image_to_handle_thresh)
    b.set(xlim=[0, 2.2], ylim=[0, 2.2], xlabel='(b)')
    
    
    f = set_nodes_coord[0,:,:]
    height = f[:,2]
    c = fig_all.add_subplot(333)
    
    plot_tsp(c, set_nodes_coord[0,:,0:2], next_batch.edges[idx], next_batch.edges_values[idx],  beam_search_edge,height)
    c.set(xlim=[0, 2.2], ylim=[0, 2.2], xlabel='(c)')
    a.set_ylabel('Distance in Km', fontsize=14)
    plt.show()
    '''
#==============================================================================
#                             Main function                                   #
#==============================================================================
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--train_filepath", type=int, default=None)
    parser.add_argument("--val_filepath", type=str, default=None)
    parser.add_argument("--test_filepath", type=str, default=None)
    parser.add_argument("--train_dataset_size",type=int,default=1000000)
    parser.add_argument("--valid_dataset_size",type=int,default=1000000)
    parser.add_argument("--testing_datset_size", type=int,default = 1000000)
    parser.add_argument("--load_best_train", type = bool,default=True)
    parser.add_argument("--load_best_test", type = bool,default=True)
    parser.add_argument("--pretrained",type=bool,default=True)
    parser.add_argument("--n_epochs",type=int,default=1)
    parser.add_argument("--beam_size",type=int,default=1024)
    parser.add_argument("--batch_size",type=int,default=1)
    opts = parser.parse_args()
    # if the filee names are not specified
    if opts.train_filepath ==None:
        opts.train_filepath = f"mmwave{opts.num_nodes}_gurobi_multi_proc.txt"
    
    if opts.val_filepath == None:
        opts.val_filepath = f"mmwave{opts.num_nodes}_val_Gurobi_multi_proc.txt"
        
    if opts.test_filepath ==  None:
        opts.test_filepath = f"mmwave{opts.num_nodes}_test_Gurobi_multi_proc.txt"
        
        
    display_step = 1000
    lr = 0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    c_lambda = 10
    crit_repeats = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_mse = []
    
    
    gen = Generator().to(device) 
    crit = Critic().to(device) 
    
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)
    with open('Model_results_summary.txt',"a" , encoding="utf-8") as f:
        
        f.write('Model Parameters')
        f.write("Number of epochs = " + str(opts.n_epochs))
        f.write("Pretrained = " + str(opts.pretrained))
        f.write("Pre-trained with best = " + str(opts.load_best_train))
        f.write("Tested with best = " + str(opts.load_best_test))
       
    
    gen,val_mse = train_model(gen,crit,opts.num_nodes,opts.batch_size,opts.train_filepath,opts.val_filepath,opts.train_dataset_size,opts.valid_dataset_size,opts.n_epochs,opts.pretrained,opts.load_best_train)
    model_testing(gen,opts.num_nodes, opts.testing_datset_size, opts.beam_size,opts.test_filepath,opts.load_best_test)
    #status = 'train'
    plot_model_results(np.asarray(val_mse),'train') 
