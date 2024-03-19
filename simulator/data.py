import h5py 



class Simulation:
    def __init__(self, simulation_output, simulation_parameters, simulation_time):
        self.output = simulation_output
        self.parameters = simulation_parameters
        self.time = simulation_time

    @classmethod
    def from_dict(self, simulation_dict):
        return Simulation(simulation_dict['traces'], simulation_dict['parameters'], simulation_dict['time'])
def save_simulation_runs(save_dir, data_name, run_id, simulation_ouputs:list, simulation_parameters:list, simulation_times:list, append=True):
    for so, sp, st in zip(simulation_ouputs, simulation_parameters, simulation_times):
        save_simulation_run(save_dir, data_name, run_id, so, sp, st, append=append)
        append = True

def save_simulation_run(save_dir, data_name, run_id, simulation_ouput, simulation_parameters, simulation_time:float, append=True):
    """
    save_dir : Directory in which the data will be saved
    data_name : Name of data file the output should be saved in
    run_id : ID of the run
    simulation_output : Output of the simulation
    simulation_parameters : Parameters used for the simulation
    simulation_time : Time of the simulation
    append : If True, the data will be appended to the file, otherwise the file will be overwritten
    """
    if type(run_id) != str:
        run_id = str(run_id)
    simulation_time = [simulation_time]
    if append: 
        file = h5py.File(save_dir + data_name + '.h5', 'a')
        # We except the .h5 file to have the following groups
        # - traces
        # - parameters
        # - time
        # We will append the data to the existing file
        file['traces'].create_dataset(run_id, data=simulation_ouput)
        file['parameters'].create_dataset(run_id, data=simulation_parameters)
        file['time'].create_dataset(run_id, data=simulation_time)

        # Then we close the file
        file.close()
    else:
        # Creat the file 
        file = h5py.File(save_dir + data_name, "w")
        # We create the groups
        file.create_group('traces')
        file.create_group('parameters')
        file.create_group('time')
        # We save the data
        file['traces'].create_dataset(run_id, data=simulation_ouput)
        file['parameters'].create_dataset(run_id, data=simulation_parameters)
        file['time'].create_dataset(run_id, data=simulation_time)
        # Then we close the file
        file.close()

def load_simulation_run(file, run_id):
    """
    Given a specific run_id this function retrievs the data from this run from the file
    """
    simulation_run = {}
    simulation_run['traces'] = file['traces'][run_id]
    simulation_run['parameters'] = file['parameters'][run_id]
    simulation_run['time'] = file['time'][run_id]
    return Simulation.from_dict(simulation_run)


def load_simulation_runs(save_dir, data_name):
    """
    This specific function is used to read a data file and extract simulation runs from it 
    """
    file = h5py.File(save_dir + data_name + '.h5', 'r')
    simulation_runs = []
    for run_id in file['traces']:
        simulation_runs.append(load_simulation_run(file, run_id))
    
    return simulation_runs