import time
import argparse
import pprint as pp
import os


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--filename", type=str, default=None)
    opts = parser.parse_args()
    
    
    if opts.filename is None:
        opts.filename =  f"Gurobi_mmwave{opts.num_nodes}_sumrate_obj_core0.txt"
    
    # Pretty print the run args
    pp.pprint(vars(opts))
    filedata = open(opts.filename, "r").readlines()
    print("Total samples: ", len(filedata))  
    test_data = filedata[opts.val_size:(opts.val_size+opts.test_size)]
    print("test samples: ", len(test_data))    
    val_data = filedata[:opts.val_size]
    print("Validation samples: ", len(val_data))
    train_data = filedata[2*opts.val_size:]
    print("Training samples: ", len(train_data))
    
    # Create separate validation data file
    with open("Gurobi_mmwave{opts.num_nodes}_val_data.txt".format(opts.num_nodes), "w", encoding="utf-8") as f:
        for line in val_data:
            f.write(line)
    
    # Create separate train data file
    with open("Gurobi_mmwave{opts.num_nodes}_train_data.txt".format(opts.num_nodes), "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line)
    
    # Create separate test data file
    with open("Gurobi_mmwave{opts.num_nodes}_test_data.txt".format(opts.num_nodes), "w", encoding="utf-8") as f:
        for line in test_data:
            f.write(line)
   
    
    