import h5py
from pathlib import Path
import logging
import argparse
from utils import is_within_bounds
import sys 
import numpy as np 
from tqdm import tqdm
logging.basicConfig(level=logging.INFO,
                        format='tuwzc1n-cortex(%(asctime)s) - %(levelname)s - %(message)s', datefmt="%H:%M:%S")
logger = logging.getLogger("collect simulations")
logger.setLevel(logging.INFO)


METRICS =              {'rate_e' : (1,50),
                       'rate_i' : (1,50),
                       'cv_isi' : (0.7, 1000),
                       'f_w-blow' : (0, 0.1),
                       'w_creep' : (0.0, 0.05),
                       'wmean_ee' : (0.0, 0.5),
                       'wmean_ie' : (0.0, 5.0),
                       'mean_fano_t' : (0.5, 2.5),
                       'mean_fano_s' : (0.5, 2.5), 
                       'auto_cov' : (0.0, 0.1),
                       'std_fr' : (0, 0.5),
                       "std_rate_spatial" : (0, 5)
                       }

def copy_item(source, dest):
    """ Recursively copy items from source to destination (both h5py Group/Dataset objects). """
    if isinstance(source, h5py.Dataset):
        # Copy dataset from source to dest
        dest.create_dataset(source.name.split('/')[-1], data=source[()])
        # Copy attributes of the dataset
        for attr_name, attr_value in source.attrs.items():
            dest[source.name.split('/')[-1]].attrs[attr_name] = attr_value
    elif isinstance(source, h5py.Group):
        # Create a new group in dest if it doesn't exist
        new_group = dest.create_group(source.name.split('/')[-1])
        # Recursively copy all items in this group
        for item_name, item in source.items():
            copy_item(item, new_group)

def merge_hdf5_files(source_dir, output_file, metrics, threshold=0.95):
    source_dir = Path(source_dir)
    total_simulations = 0
    valid_simulations = 0
    import os
    if os.path.exists(output_file):
        mode = "a"
    else:
        mode = "w"
    logger.info(f" Searching for result files (mode: {mode})")
    with h5py.File(output_file, mode) as output_hdf5:
        # Iterate over all HDF5 files in the source directory
        logger.info(f"Found {len(list(source_dir.glob('*.hdf5')))} raw result files")
        for file_path in tqdm(source_dir.glob('*.hdf5')):

            # Open the file
            with h5py.File(file_path, 'r') as input_hdf5:
                group_name = str(file_path.stem)
                if group_name in output_hdf5:
                    group = output_hdf5[group_name]
                else:
                    group = output_hdf5.create_group(group_name)
                for item_name, item in input_hdf5.items():
                    if item_name not in group:
                        input_hdf5.copy(item, group, name=item_name)  # Copy item into the correct group
                # Check if this group is valid according to defined metrics
                
                if metrics: 
    
                    is_valid = True
                    for metric in metrics: 
                        valid = is_within_bounds(group, metric, METRICS[metric])
                        if not valid:
                            is_valid = False
                        
                    if is_valid: 
                        valid_simulations += 1
            total_simulations += 1
            file_path.unlink()
    
   
    logger.info(f"All data has been merged into {output_file} - collected {total_simulations}")
    if metrics:
       
        logger.info(f"Found (valid / total){valid_simulations} / {total_simulations}  in current batch ")
        relative_amount = valid_simulations / total_simulations if total_simulations > 0 else 0
        if relative_amount < threshold:
            logger.info("Below threshold - continuing with same metrics")
            sys.exit(1)
    sys.exit(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect simulations.")
    parser.add_argument('--metrics', metavar='metrics', type=str,
                                help='list of metric categories to be applied', default="")
    parser.add_argument("--working_dir", type=str)
    args = parser.parse_args()
    metrics = args.metrics
    metrics = metrics.split()
    working_dir = args.working_dir
    temp_sim_runs = Path(working_dir) / "raw_results"
    output_file = Path(working_dir) / "simulations.hdf5"
    
    merge_hdf5_files(temp_sim_runs, output_file, metrics=metrics)
