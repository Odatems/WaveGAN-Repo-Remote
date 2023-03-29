import time
import argparse
import pprint as pp
import os



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cores", type=int, default=20)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--base_filename", type=int, default=None)
    parser.add_argument("--filename", type=str, default=None)
    opts = parser.parse_args()
    
    
    if opts.filename is None:# to write for 
        opts.filename =  f"Gurobi_mmwave{opts.num_nodes}_sumrate_obj.txt"
        
    if opts.base_filename is None:# The based file to read from 
        opts.base_filename = f"Gurobi_mmwave{opts.num_nodes}_sumrate_obj_core"
        
        
        
        
    for i in range(opts.num_cores):
        
        filedata = open(opts.base_filename+"{}.txt".format(i+1), "r").readlines() # Read files
        with open(opts.filename, "a", encoding="utf-8") as f: # Append to file
            for line in filedata:
                f.write(line) 
                
    


