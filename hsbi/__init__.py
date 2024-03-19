"""
Hypothesis-SBI should work in the following way:

- A metric can be declared as broad, then all specified parameters in the model will 
- be varied to find feasible combinations of parameters that satify this metric

- A metric can be declared as specific, then it is bounded to specific parameters in the 
- model and the search for feasible combinations of parameters is limited to these parameters


In the estimation of the posterior p(θ|x) we will use the prinicple of fSBI: 
- We will first search for feasible combinations of parameters that satisfy broad metric (in fSBI) manner
- We will then use the trained posterior p(θ|x) as the prior for the ongoing search
- A second posterior p2(θ*|x) will be t rained onthe specific metric while other parameters are fixed (fixed in the sense
  that they will be sampled according to the prior p(θ|x))

# Example 

We have metrics that specify stable network dynamics such as
- The firing rate of the excitatory population should be in the range of 5-10 Hz
- The firing rate of the inhibitory population should be in the range of 5-10 Hz
etc. 

Such metrics can be specified as broad metrics, because each parameter in our network should 
play a role in satisfying these metrics. 

We then have task specific metrics e.g. a delayed memory task, in heres comes the hypothesis into play -
we hypothesise that a specific neuronal population or connection type e.g. I-to-I is responsible for the
functionality of that task in the circuit. 

By defining this metric as specific we are limiting the search for feasible combinations of parameters 
that we define as relevant for that parameters, while at the same time only searching in the overall subspace
of parameters that result in stable network dynamics (broad metrics).

Each parameter in the model should be specified by an identifier.

When we fit the posterior p(θ|x) we initally fit









"""
from concurrent import futures 

class HSBI: 
    
    def __init__(self, num_workers=10):
        self.num_workers = 10
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