# WaveGAN
1. **To generate dataset using Gurobi, run any of the following commands based on the objective:**
      
      
      `python Gurobi_mmwave_sumrate_generator.py --num_nodes {Number of Nodes} --num_samples {dataset size} --node_dim {3d coordinates} --num_cores {Number of the cores for multiprocessing} --max_h {maximum height} --max_y {y-limit of the deployment region} --max_x {x-limit of the deployment region} --fun_type {}`

      ***Where: --fun_type is 1 for *sum-rate* objective and 2 for *log-sum* objective***

