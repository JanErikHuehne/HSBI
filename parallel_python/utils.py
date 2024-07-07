import h5py
import numpy as np 
import logging
import torch as th 
from tqdm import tqdm
logging.basicConfig(level=logging.INFO,
                        format='MAIN tuwzc1n-cortex(%(asctime)s) - %(levelname)s - %(message)s', datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Decorator to ensure function returns a torch tensor
def return_tensor(func):
    def wrapper(*args, **kwargs):
        func_result = func(*args, **kwargs)
        if type(func_result) == tuple:
            return (th.tensor(i) for i in func_result)
        else:
            return th.tensor(func_result)
    return wrapper

def print_hdf5_structure(group, indent=0):
    """
    Recursively prints the structure and contents of an HDF5 group.
    :param group: The HDF5 group or file to print.
    :param indent: The current indentation level for pretty printing.
    """
    spacing = ' ' * indent
    for name, item in tqdm(group.items()):
        if isinstance(item, h5py.Group):
            # It's a group; print its name and continue recursively
            print(f"{spacing}Group: {name}/")
            print_hdf5_structure(item, indent + 4)
        elif isinstance(item, h5py.Dataset):
            # It's a dataset; print its name and contents (optional)
            print(f"{spacing}Dataset: {name}, shape: {item.shape}, dtype: {item.dtype}")
            # If you want to print contents, be careful with large datasets
            try:
                print(f"{spacing}  Contents: {item[()]}")
            except Exception as e:
                print(f"{spacing}  (Error reading dataset contents: {e})")

def is_within_bounds(group, metric, bounds):
    """
    Checks if the metric in a given group is within the specified bounds.
    :param group: HDF5 group object containing datasets.
    :param metric: Metric name as a string.
    :param bounds: Tuple of (lower_bound, upper_bound).
    :return: True if the metric is within bounds, False otherwise.
    """
    if metric in group.keys():
        metric_value = float(group[metric][()])
        lower_bound, upper_bound = bounds
        return_result =  lower_bound <= metric_value <= upper_bound
        return return_result
    
    return False

@return_tensor
def collect_simulations(metrics, metrics_bounds, h5_path):
    """This function is used to collect filtered simulation results. 

    Args:
        metrics (list): list of string metrics to constrain simulations
        metrics_bounds (dict): mapping metrics to metrics_bounds (tuple (low high))
        h5_path (str or pathlib.Path): path to the simulation storage file (.hdf5 format)

    Returns:
        tuple: (sim_parameters, observations)
    """
    parameter_lists = []
    observed_metrics = []
    with h5py.File(h5_path, 'r') as file:
        for _, group in tqdm(file.items()):
            if all(is_within_bounds(group, metric, metrics_bounds[metric]) for metric in metrics if metric in metrics_bounds):
                metrics_data = [float(group[metric][()]) for metric in metrics if metric in group]
                observed_metrics.append(metrics_data)
                if 'run_parameters' in group:
                    parameters_data = eval(group['run_parameters'][()])
                    parameter_lists.append(parameters_data)
        return (parameter_lists, observed_metrics)


import h5py

def list_all_groups_and_datasets(group, base_path="/"):
    """
    Recursively lists all groups and datasets in an HDF5 group.
    :param group: The HDF5 group to start with (can be the root file object).
    :param base_path: The base path for the current group (used for recursion).
    :return: A tuple containing two lists - groups and datasets.
    """
    groups = []
    datasets = []
    
    for name, item in tqdm(group.items()):
        item_path = f"{base_path}{name}/"
        if isinstance(item, h5py.Group):
            groups.append(item_path)
            subgroups, subdatasets = list_all_groups_and_datasets(item, item_path)
            groups.extend(subgroups)
            datasets.extend(subdatasets)
        elif isinstance(item, h5py.Dataset):
            datasets.append(item_path.rstrip('/'))
    
    return groups, datasets

# Usage Example
# Replace 'your_file.h5' with the path to your HDF5 file
#with h5py.File('/home/ge84yes/master_thesis/HSBI/data/hsbi_data/simulations.hdf5', 'r') as hdf5_file:
#    all_groups, all_datasets = list_all_groups_and_datasets(hdf5_file['2fbef958853017841ca46f188c628c7b65ff49625ec6ef3177638c4a97346caf'])
#    print("Groups:")
#    for group in all_groups:
#        print(group)
#    print("\nDatasets:")
#    for dataset in all_datasets:
#        print(dataset)


