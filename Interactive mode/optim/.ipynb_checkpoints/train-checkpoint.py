import numpy as np
import torch
from torch import nn
from models.Descriminator import *
from models.Generator import *


def train_model(gen,crit,train_filepath,val_filepath,train_dataset_size,valid_dataset_size,n_epochs,pretrained=False,load_best=True):
  
    '''
    train_filepath: is the file containig training dataset
    val_filepath: is the file containig validation dataset
    n_epochs: Number of epochs to train for 
    pretrained: if pretrained model to be used
    load_best: if oretrained: either best model trained so far or the recent one.
    '''
    
    X_train, z_norm,validation_set_sample,z_norm_valid  = load_dataset(train_filepath,val_filepath,num_nodes,train_dataset_size, valid_dataset_size)
    
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
        
        
        
    start = time.time()   
    mse_best = 100 # best mse encountered so far in case to continue training
    # training loop over all dataset
    for epoch in range(scale_factor*n_epochs):
        torch.cuda.empty_cache()
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
               
               tick_width = 50
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
        mse_val.append( mse_per_inst_val[epoch,:].mean())
        
        print(f'Epoch {i}/{n_epoch}:----training mse = , validation mse = {mse_val[epoch]}, Generator loss: {gen_mean}, critic loss: {crit_mean}')
        
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
       
       f.write('Training Results statistics')
       f.write("Number of nodes = " + str(num_nodes))
       f.write("Number of training samples = " + str(testing_datset_size))
       f.write("Number of validation samples = " + str(valid_dataset_size))
       f.write("training time = " + str(end_time))
       f.write("best mse found = " + str(mse_best))
       f.write("Vlidation mean square error = ", str(mse_val) ) # store the validation mean square error 
       
    return gen, mse_val