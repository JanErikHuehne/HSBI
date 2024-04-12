import h5py 
import os
import numpy as np 
import json 
import logging
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class SimulationHDF5:
    def __init__(self, save_dir="./data", **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        self.filename = save_dir + "/results.h5"
    
    def save_simulation(self, parameters, outputs):
        with h5py.File(self.filename, 'a') as f:
            if 'simulations' not in f:
                f.create_group('simulations')

            sim_group = f['simulations']
            sim_idx = len(sim_group)
            sim_group.create_group(f'sim_{sim_idx}')
            sim_group[f'sim_{sim_idx}']['parameters'] = json.dumps(parameters)
            sim_group[f'sim_{sim_idx}']['outputs'] = np.array(outputs)
        
    def save_simulations(self, parameter_set, output_set):
        for p, o in zip(parameter_set, output_set):
            self.save_simulation(p,o)
    
    def get_all_simulations(self):
        simulations = []
        with h5py.File(self.filename, 'r') as f:
            if 'simulations' in f:
                sim_group = f['simulations']
                for sim_name in sim_group:
                    parameters = json.loads(sim_group[sim_name]['parameters'][()])
                    outputs = sim_group[sim_name]['outputs'][:]
                    simulations.append((parameters, outputs))
        return simulations
    
    def get_filtered_simulation(self, metrics, metric_order):
        """
        This function retrieves all the simulations, filters and strips them according to the metrics provdided before returning 
        a tuple of (parameters, outputs) as np.arrays. 
        """
        
        simulations = self.get_all_simulations()
        thetas = []
        obs = []

        # First we map metric names to index in the output of the simulation
        ids = []
        for metric in metrics: 
            index = metric_order.index(metric[0])
            ids.append(index)
        for sim in simulations:
            o_out = sim[1]
            # We iterate over all metrics to filter 
            
            for metric, id in zip(metrics,ids):
                # get the respective metric index in the output
                # Check if output is out of [low_bound, up_bound] of the metric 
                if o_out[id] < metric[1][0] or o_out[id] > metric[1][1]:
                    # Remove this simulation and break (continue with next simulation sample)
                    logger.debug(f'Removing simulation violating {metric[0]}([{ metric[1][0]}, { metric[1][1]}]) with value {o_out[id]}')
                    simulations.remove(sim)
                    break
        logger.debug(f"{len(simulations)} Simulations remaining after filtering")
    
      

        # Now we will get the format to return the simulations as pair of numpy arrays
        thetas = []
        obs = []
        for sim in simulations:
            th_raw = sim[0]['parameters']
            th = []
            for s in list(th_raw.keys()):
            
                th= th + list(th_raw[s].values())
            
            out = sim[1]
            # only select ids of the passed metrics
            out = out[ids]
            thetas.append(th)
            obs.append(out)
        return np.array(thetas).astype(np.float32), np.array(obs).astype(np.float32)
       
        





            