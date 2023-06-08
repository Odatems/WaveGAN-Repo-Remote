# This code is to solve mmwave problem using Concorde code for matricies

import numpy as np
from itertools import combinations
import time
from concorde.tsp import TSPSolver
import os
import multiprocessing as mp
from concorde.tsp import TSPSolver
import argparse

np.random.seed(1)

num_odes = 20
def solve_model(input_data):#,*args):
    
    print(input_data.shape)
    fun_type = 2
    
    core_id = os.getpid()
    num_nodes = 20
    set_nodes_coord = input_data
    #fun_type,core_id,num_nodes,set_nodes_coord = input_data
    solver_pid = TSPSolver()
   
    # Step 1: Generate random coordnates to represent the network
    node_dim = 3 #  3d coordinates
    num_samples = 1#set_nodes_coord.shape[0]#5000 # dataset size

    nodes = [i for i in range(num_nodes)]
    coeff_matrix = np.zeros((num_nodes,num_nodes)) # create coeffecient matrix to be passed to the solver
   
   
    # -------------Network related parameters ------------------------------
    
    
    p_fso = 10**(-3)
    N_b = 67885
    C=299792458    # light speed
    k_planck_const = 6.626*10(-34)
    lamda = 1550*10**(-9)
    E_p = k_planck_const*C/lamda
    gamma = 10**(1/10)*10**3 # I converted from km to meter. 
    omega = 42.5 *10**(-3)
    tau_fso_tx = 0.9
    tau_fso_rx = 0.7
    epsilon = 10**-3
    
    if fun_type == 1:
        filename = f"Concorde_mmwave{num_nodes}_sumrate_obj_core{core_id}.txt"
    elif fun_type == 2:
        filename = f"Concorde_mmwave{num_nodes}_log_sum_obj_core{core_id}.txt"
   
    for i in range(num_samples):
       
        # iterte over the combinations of nodes for the current graph
        for j in combinations(nodes,2):
           
            current_combination = j
            tx_node = current_combination[0] # extract the sender
            rx_node = current_combination[1] # receiver coordinates
           
            #==========================================================================================================
            #X = set_nodes_coord[i,tx_node,:]
            X = set_nodes_coord[tx_node,:]
            Y = set_nodes_coord[rx_node,:]
            #Y = set_nodes_coord[i,rx_node,:]
            
            L = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2+(X[2]-Y[2])**2) # Euclidan distance
            rate_values = (p_fso * tau_fso_tx*tau_fso_rx*(10**(-gamma*L/10)*omega))/(np.pi*(epsilon/2)**2 * L**2*E_p*N_b)
            
            D = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2+(X[2]-Y[2])**2) # Euclidan distance
           
            #===================================End of the channel model================================================
            coeff_matrix[tx_node,rx_node] = rate_values
            coeff_matrix[rx_node,tx_node] = rate_values
            #print(coeff_matrix[rx_node,tx_node])
        # normalize between 0 and 1
        coeff_matrix_normalized = (coeff_matrix - np.min(coeff_matrix))/(np.max(coeff_matrix) - np.min(coeff_matrix))#*100  
        
        #dist =  -rate_values
        print('hello world')
        if fun_type == 1:  
            solution = solver_pid.solve_mat(-rate_values,num_nodes)
        elif fun_type == 2:
         pos = np.where(rate_values == 0)
         rate_values[pos] = 1e-5
         val = np.log(rate_values)
         
         
         solution = solver_pid.solve_mat(-val,num_nodes)
           
        # Step 4: Write the cost matrix and the solution to text file
        with open(filename, "a") as f:
           
            f.write( " ".join( str(coeff_matrix_normalized[row_idx,col_idx])+str(" ") for row_idx,col_idx in combinations(nodes, 2)))
            f.write( str(" ") + str('output') + str(" ") )
            f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour))
            f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
            f.write( "\n" )
        f.close()
       
       
#------------------------------------------------------------------------------

if __name__ == '__main__':
 
    # Generate random coordinates
    min_x = 0
    min_y = 0
    min_h = 0
   
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_cores", type=int, default=20)
    parser.add_argument("--node_dim", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=1000000)
    parser.add_argument("--max_x", type=int, default=500)
    parser.add_argument("--max_y", type=int, default=500)
    parser.add_argument("--max_h", type=int, default=200)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--fun_type", type=int, default=2)
    opts = parser.parse_args()
   
   
    data_per_core = np.int16(np.floor(opts.num_samples/opts.num_cores))
    set_nodes_coord = np.random.random([opts.num_samples, opts.num_nodes, opts.node_dim])*100
    set_nodes_coord = np.random.randint(min_h, high=opts.max_h, size=(opts.num_samples, opts.num_nodes, opts.node_dim))
    set_nodes_coord[:,:,1] = np.random.randint(min_y, high=opts.max_y, size=(opts.num_samples, opts.num_nodes))
    set_nodes_coord[:,:,0] = np.random.randint(min_x, high=opts.max_x, size=(opts.num_samples, opts.num_nodes))
   
       
    set_nodes_coord[:,:,2] = np.random.randint(min_h, high=opts.max_h, size=(opts.num_samples, opts.num_nodes))
    yy = np.where(set_nodes_coord[:,:,2]<min_h) # find the coordinates where the height of the uav is less 20
    set_nodes_coord[yy[0],yy[1],2] = set_nodes_coord[yy[0],yy[1],2] + min_h
    
    procs = []
    shift_val = 0
    
    with mp.Pool(20) as pool:     
     pool.map(solve_model,set_nodes_coord,chunksize=1)
	
    #solve_model(set_nodes_coord)	
    
    '''
    for i in range(opts.num_cores):
       
       
        process = mp.Process(target=solve_model,args= [(opts.fun_type,i,opts.num_nodes,set_nodes_coord[i*data_per_core:(i+1)*data_per_core])])
        procs.append(process)
        process.start()
   
    for proc in procs:
        proc.join()
    '''
