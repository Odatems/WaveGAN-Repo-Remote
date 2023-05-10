# This code is to solve mmwave problem using Concorde code for matricies 

import numpy as np
from itertools import combinations 
import time
from concorde.tsp import TSPSolver
import os
import multiprocessing as mp
from concorde.tsp import TSPSolver
import argparse




def solve_model(input_data,*args):
    
    
    fun_type,i,num_nodes,set_nodes_coord = input_data
    solver_pid = TSPSolver()
    
    # Step 1: Generate random coordnates to represent the network 
    node_dim = 3 #  3d coordinates
    num_samples = set_nodes_coord.shape[0]#5000 # dataset size

    nodes = [i for i in range(num_nodes)]
    coeff_matrix = np.zeros((num_nodes,num_nodes)) # create coeffecient matrix to be passed to the solver 
    
    
    # -------------Network related parameters ------------------------------
    n=2            #pathloss exponent
    f_c=30*1e9     # carrier frequency (30GHz)
    C=299792458    # light speed
    N0=10**(-204/10) # Thermal Noise
    alpha=9.6        # probablity LoS parameter
    beta=0.28       # probablity LoS parameter
    Rainrate=50     # rain rate in mm/h
    XLos=1         #dB LoS loss
    
    kH=0.8606   
    kV=0.8515 
    alphaH=0.7656 
    alphaV=0.7486
    tau=45 #% tilt angle relative to the horizantal 45 degree 
    theta=0 #%path elevation angle (Tx and Rx at different altitude) 
    BandwidthmmWave = 500*1e6 
    GainX = 5011.87
    GainY=5011.87
    bandwidth = 500*1e6 
    power = 4

    if fun_type == 1:
        filename = f"Concorde_mmwave{num_nodes}_sumrate_obj_core{i}.txt"
    elif fun_type == 2:
        filename = f"Concorde_mmwave{num_nodes}_sumrate_obj_core{i}.txt"
    
    for i in range(num_samples): 
        
        # iterte over the combinations of nodes for the current graph
        for j in combinations(nodes,2):
            
            current_combination = j
            tx_node = current_combination[0] # extract the sender
            rx_node = current_combination[1] # receiver coordinates
            
            #==========================================================================================================
            X = set_nodes_coord[i,tx_node,:]
            Y = set_nodes_coord[i,rx_node,:]
            
            
            D = np.sqrt((X[0]-Y[0])**2+(X[1]-Y[1])**2+(X[2]-Y[2])**2) # Euclidan distance
            # Compute loss due to environmental factors
            
            LAtt=14.25*D/1000+0.0162; #dB att. due to vapor water  LAtt=Lvap+LO2 https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.676-11-201609-I!!PDF-E.pdf obtained from the curve Fig. 2_energy_efficient Los V
            
            k=1/2*(kH+kV+(kH-kV)*(np.cos(theta))**2*np.cos(2*tau))   #%
            alpha=1/(2*k)*(kH*alphaH+kV*alphaV+(kH*alphaH-kV*alphaV)*(np.cos(theta))**2*np.cos(2*tau))
            LR=k*Rainrate**alpha #% att. rain attenuation 
            PLd=LAtt+D/1000*LR # %dB 
            
            
            theta=np.arcsin(np.absolute(X[2]-Y[2])/D) # angle
            thetadegree=180*theta/np.pi # conversion of the angle
            PLLoS=10*n*np.log10(4*np.pi*f_c*D/C)+XLos # pathloss
            Prob=1/(1+alpha*np.exp(-beta*(thetadegree-alpha))) # probability of existence of LoS link
            
            if (X[2]<=20 and Y[2]<=20) or (((X[2]<=20 and Y[2]>20) or (X[2]>20 and Y[2]<=20)) and Prob<0.9) or (D>500):
                #CNRmmWave=0 # No transmission if tranceivers have low altitude or LoS prob is less than 0.9 or long distance
                CNRmmWave=1e-30
            else:
                PLdB=PLLoS+PLd; # Loss = Pathloss + atmospheric loss
                PLA2G=10**(0.1*PLdB)
                CNRmmWave=(GainX*GainY*(1/np.sqrt(PLA2G))**2)/(N0*BandwidthmmWave) #channel to noise ratio
            
            #============================================================================================================
            coeff_matrix[tx_node,rx_node] = CNRmmWave
            coeff_matrix[rx_node,tx_node] = CNRmmWave
            #print(coeff_matrix[rx_node,tx_node])
        
        # normalize between 0 and 1
        coeff_matrix_normalized = (coeff_matrix - np.min(coeff_matrix))/(np.max(coeff_matrix) - np.min(coeff_matrix))#*100   
        rate_values = bandwidth*np.log2(1+power*coeff_matrix_normalized)*1e-8
        #dist =  -rate_values
        if fun_type == 1:   
            solution = solver_pid.solve_mat(-rate_values,num_nodes) 
        elif fun_type == 2:
            solution = solver_pid.solve_mat(-np.log(rate_values),num_nodes) 
            
        # Step 4: Write the cost matrix and the solution to text file 
        with open(filename, "a") as f:
            
            f.write( " ".join( str(coeff_matrix_normalized[row_idx,col_idx])+str(" ") for row_idx,col_idx in combinations(nodes, 2)))
            f.write( str(" ") + str('output') + str(" ") )
            f.write( str(" ").join( str(node_idx+1) for node_idx in solution.tour))
            f.write( str(" ") + str(solution.tour[0]+1) + str(" ") )
            f.write( "\n" )
    
    
    
#------------------------------------------------------------------------------

if __name__ == '__main__':
  
    # Generate random coordinates
    min_x = 0
    min_y = 0 
    min_h = 0 
   
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_cores", type=int, default=10)
    parser.add_argument("--node_dim", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=10e6)
    parser.add_argument("--max_x", type=int, default=500)
    parser.add_argument("--max_y", type=int, default=500)
    parser.add_argument("--max_h", type=int, default=200)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--fun_type", type=int, default=1)
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
    
    for i in range(opts.num_cores):
        
        
        process = mp.Process(target=solve_model,args= [(opts.fun_type,i,opts.num_nodes,set_nodes_coord[i*data_per_core:(i+1)*data_per_core])])
        procs.append(process)
        process.start()
   
    for proc in procs:
        proc.join()
    
