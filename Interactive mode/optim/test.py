import torch 
import numpy as np 
from torch import pytorch
from models.Descriminator import *
from models.Generator import Generator


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
        xxx_test = xxx_valid.to(device)       
        
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
        
        optimality_gap_in_thr[i] =  np.absolute(1 - (rate_values[cur_node_to_investigate+shift_index]/rate_values_gen[i])) 
        optimality_gap_in_thr[i] = optimality_gap_in_thr[i]*100
       
    end_time =  time.time() - start
    device = 'cuda'
    print(f'End of the testing phase; Optimality gap in cost values = {np.mean(optimality_gap_in_cost)}, Optimality gap in throughput = {np.mean(optimality_gap_in_thr)}')
    
    with open('Model_results_summary.txt',"a" , encoding="utf-8") as f:
       
       f.write('Test results statistics')
       f.write("Number of nodes = " + str(num_nodes))
       f.write("Number od testing samples = " + str(testing_datset_size))
       f.write("Beam size = " + str(beam_size))
       f.write("Testing time = " + str(end_time))
       f.write("Optimality gap in throughput = ", str(np.mean(optimality_gap_in_thr)))
       f.write("Optimality gap in cost matrix values = ", str((np.mean(optimality_gap_in_cost)))) # this might be different because of the differences in the cost matrix 
       f.write("Final netowrk topology = ", str(final_tour)) # the final tour generated after beam search
