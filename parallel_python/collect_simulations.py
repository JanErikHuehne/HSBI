import h5py
from pathlib import Path
import logging
import argparse
logger = logging.logger(__name__)

def merge_hdf5_files(source_dir, output_file):
    source_dir = Path(source_dir)
    with h5py.File(output_file, "a") as output_hdf5:
        # Iterate over all HDF5 files in the source directory
        for file_path in source_dir.glob('*.hdf5'):
            # Open the file
            with h5py.File(file_path, 'r') as input_hdf5:
                group_name = file_path.stem
                group = output_hdf5.create_group(group_name)
                for ds_name, dataset in input_hdf5.items():
                    # Create dataset in output 
                    group.create_dataset(ds_name, data=dataset[()])
                    # Optionally, you can copy attributes if needed:
                    for attr_name, attr_value in dataset.attrs.items():
                        group[ds_name].attrs[attr_name] = attr_value
            file_path.unlink()
    logger.info(f"All data has been merged into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect simulation results")
    parser.add_argument("--working_dir", type=str)
    args = parser.parse_args()
    temp_sim_runs = Path(args.working_dir) / "raw_results"
    output_file = Path(args.working_dir) / "simulations.hdf5"
    merge_hdf5_files(temp_sim_runs, output_file)
