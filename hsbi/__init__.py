from concurrent import futures 

class HSBI: 
    
    def __init__(self, num_workers=10):
        self.num_workers = num_workers
        self.simulator = None
    def sim_run(self, *args, **kwargs):
        """
        This function calls the simulation for a single run, 
        and returns the output of the simulation.
        """
      
        ###################################################
        ### 
        ###
        ### Here we simulation call needs to be implemented
        ###
        ###
        ###################################################
        return None
    
    def simulate(self, sim_params:list, seeds:list, *args, **kwargs):
        """
        With this function we want to simulate 
        """
        print('Simulating HSBI')
        job_inputs = [dict(param=th, seeds=[seed]) for th,seed in zip(sim_params, seeds)]
        with futures.ThreadPoolExecutor(max_workers=self.num_workers) as exectuor:
            jobs = [exectuor.submit(self.simulate, inp) for inp in job_inputs]
        outputs = []
        for job in jobs: 
            outputs.append(job.result())
        return outputs

    def fit_posterior(self, *args, **kwargs):
        """
        This function fits the posterior of the model
        """
        pass