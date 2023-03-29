
import numpy as np
from itertools import combinations 
import time
# Define Gurobi solver 
import gurobipy as gp
from gurobipy import GRB
import os



#================ Gurobi Solver common functions ==============================

# This function is based on the code published on the main page of gurobi 
#https://www.gurobi.com/jupyter_models/traveling-salesman/


def subtourelim(model, where):
    
    
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                             if vals[i, j] > 0.5)
        node_id = gp.tuplelist((i, j) for i, j in model._vars.keys())
        num_nodes = np.int16(np.ceil(np.sqrt(len(node_id))))
       
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        nodes = [i for i in range(num_nodes)]
        if len(tour) < len(nodes):
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in combinations(tour, 2))
                         <= len(tour)-1)

# Given a tuplelist of edges, find the shortest subtour

def subtour(edges,num_nodes=20):
    nodes = [i for i in range(num_nodes)]
    unvisited = nodes[:]
    cycle = nodes[:] # Dummy - guaranteed to be replaced
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(thiscycle) <= len(cycle):
            cycle = thiscycle # New shortest subtour
    return cycle



#------------------ The following function solves the model -----------------


def solve_model(input_data,*args):
    
    
    with gp.Env() as env, gp.Model(env=env) as m: 
    
        fun_type,core_id,num_nodes, set_nodes_coord = input_data
        num_samples = set_nodes_coord.shape[0]
        
        if(fun_type == 1): # sum-rate
            filename =  f"Gurobi_mmwave{num_nodes}_sumrate_obj_core{core_id}.txt"
        elif(funtype==2): # log-sum
            filename =  f"Gurobi_mmwave{num_nodes}_logsum_obj_core{core_id}.txt"
        
        # Step 1: Define the model 
        m = gp.Model() # tested with Python 3.7 & Gurobi 9.0.0
        
        # Step 2: Solve the problem
        # Step 1: Generate random coordnates to represent the network 
        node_dim = 3 # we work in 3d  
        nodes = [i for i in range(num_nodes)]
        coeff_matrix = np.zeros((num_nodes,num_nodes))    
        
        # Network related parameters 
        n=2    #pathloss exponent
        f_c=30*1e9     # carrier frequency (30GHz)
        C=299792458    # light speed
        N0=10**(-204/10) # Thermal Noise
        alpha=9.6 # probablity LoS parameter
        beta=0.28 # probablity LoS parameter
        Rainrate=50 # rain rate in mm/h
        XLos=1    #dB LoS loss
         
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
        
       
        for i in range(num_samples): # num_samples
            
            
            # iterte over the combinations of nodes for the current graph
            for j in combinations(nodes,2):
                
                current_combination = j
                tx_node = current_combination[0] # extract the sender
                rx_node = current_combination[1] # receiver coordinates
                
                
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
                    CNRmmWave=0 # No transmission if tranceivers have low altitude or LoS prob is less than 0.9 or long distance
                    #CNRmmWave=1e-10
                else:
                    PLdB=PLLoS+PLd; # Loss = Pathloss + atmospheric loss
                    PLA2G=10**(0.1*PLdB)
                    CNRmmWave=(GainX*GainY*(1/np.sqrt(PLA2G))**2)/(N0*BandwidthmmWave) #channel to noise ratio
                
                #============================================================================================================
                coeff_matrix[tx_node,rx_node] = CNRmmWave 
                coeff_matrix[rx_node,tx_node] = CNRmmWave 
               
            
                        
            # normalize between 0 and 1
            coeff_matrix_normalized = (coeff_matrix - np.min(coeff_matrix))/(np.max(coeff_matrix) - np.min(coeff_matrix))#*100
           
            rate_values = bandwidth*np.log2(1+power*coeff_matrix_normalized)*1e-8
            if(fun_type==1): # sum-rate
                dist = {(tx_node, rx_node):  -rate_values[tx_node,rx_node] for tx_node, rx_node in combinations(nodes, 2)} # create dictionary of variables
            elif(fun_type == 2): # sum-log
                dist = {(tx_node, rx_node):  -np.log(rate_values[tx_node,rx_node]) for tx_node, rx_node in combinations(nodes, 2)} # create dictionary of variables
                
            
            # Step 3: Solve using Gurobi Solver. 
           
            # Variables: is city 'i' adjacent to city 'j' on the tour?
            vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
            # Symmetric direction: Copy the object
            for ix, jx in vars.keys():
                vars[jx, ix] = vars[ix, jx]  # edge in opposite direction
        
            # Constraints: two edges incident to each city
            cons = m.addConstrs(vars.sum(c, '*') == 2 for c in nodes)
            # Solve the model 
            m._vars = vars
            
            m.Params.lazyConstraints = 1
            m.optimize(subtourelim)
            
           
            # Retrieve solution
        
            vals = m.getAttr('x', vars)
            selected = gp.tuplelist((ix, jx) for ix, jx in vals.keys() if vals[ix, jx] > 0.5)
        
            tour = subtour(selected,num_nodes)
            assert len(tour) == len(nodes)
        
        
            # Step 4: Write the cost matrix and the solution to text file 
            with open(filename, "a") as f:
                
                f.write( " ".join( str(coeff_matrix_normalized[row_idx,col_idx])+str(" ") for row_idx,col_idx in combinations(nodes, 2)))
                
                #f.write( " ".join( str(edge_cost)+str(" ") for edge_cost in coeff_matrix_normalized))
                f.write( str(" ") + str('output') + str(" ") )
                f.write( str(" ").join( str(node_idx+1) for node_idx in tour) )
                f.write( str(" ") + str(tour[0]+1) + str(" ") )
                f.write( "\n" )
    

#----------------------------------------------------------------------------------------------------------------------------------------------------------

import multiprocessing as mp
import gurobipy as gp
import argparse


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--node_dim", type=int, default=3)
    parser.add_argument("--num_cores", type=int, default=1)
    parser.add_argument("--max_h", type=int, default=500)
    parser.add_argument("--max_y", type=int, default=500)
    parser.add_argument("--max_x", type=int, default=500)
    parser.add_argument("--fun_type", type=int, default=1)
    opts = parser.parse_args()
    
    
    # Generate random coordinates
    min_x = 0  
    min_y = 0  
    min_h = 0 
    procs = [] 
    shift_val = 0
    data_per_core =  np.int16(np.floor(opts.num_samples/opts.num_nodes))
    
   
    set_nodes_coord = np.random.random([opts.num_samples, opts.num_nodes, opts.node_dim])*100
    set_nodes_coord = np.random.randint(min_h, high=opts.max_h, size=(opts.num_samples, opts.num_nodes, opts.node_dim))
    set_nodes_coord[:,:,1] = np.random.randint(min_y, high=opts.max_y, size=(opts.num_samples, opts.num_nodes))
    set_nodes_coord[:,:,0] = np.random.randint(min_x, high=opts.max_x, size=(opts.num_samples, opts.num_nodes))
   
  
    for i in range(opts.num_cores):
        
        
        process = mp.Process(target=solve_model,args= [(opts.fun_type,i,opts.num_nodes,set_nodes_coord[(i*data_per_core)+shift_val:((i+1)*data_per_core)+shift_val])])
        procs.append(process)
        process.start()
   
    for proc in procs:
        proc.join()
        
      
    