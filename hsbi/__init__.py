from concurrent import futures 
from .simulator.network import Rate_STDP_Network, plasticity_parameters, run_simulation
from .simulator.data import SimulationHDF5
from .density_estimator import MakePosterior
from sbi.utils import BoxUniform
from .metrics import Metrics
import numpy as np 
import logging
import torch
import hashlib
from tqdm import tqdm 
import sys
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class HSBI: 
    def __init__(self, num_workers=10, num_ensamble=3):
        self.num_workers = num_workers
        self.num_ensemble = num_ensamble
        self.simulator = None
        self.metrics_defined = False
        self.parameters_defined = False

    def define_metrics(self, metrics:list=None, metric_value:dict=None,  **kwargs):
        """
        Method to define the metrics of the run. 

        Args:
            metrics (list, optional): A list of metrics to be used. Defaults to None.
            metric_value (dict, optional): A dictionary specifying specific metric bounds. Defaults to None.

        Returns:
            None

        Raises:
            None

        Examples:
            # Define all metrics
            define_metrics()

            # Define specific metrics
            define_metrics(metrics=['cv_isi', 'spatial_Fano'])

            # Define metric bounds
            define_metrics(metric_value={'rate_e': (1, 50)})
        """
       
        self.metrics_list = {'rate_e' : (0,50),
                       'rate_i' : (0,50),
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
        # We set the metrics to the subset of metrics if they have been specified 
        if metrics is not None:
            logger.info("Found metrics")
            new_metrics = {}
            for metric in metrics:
                new_metrics[metric] = self.metrics_list[metric]
            self.metrics_list = new_metrics

        # We set the values of metrics to explicitly specified values
        if metric_value is not None:
            for metric, value in metric_value.items():
                try:    
                    assert self.metrics_list[metric]
                    self.metrics_list[metric] = value
                except AssertionError:
                    logger.warning('You specified a metric value for a metric that is not supported. Skipping... Please make sure to specify a valid metric name.')
                    continue
        self.metrics_defined = True

    def get_metrics_list(self):
        """Returns the metrics list used as a list of tuples (name, bounds)"""
        if not self.metrics_defined:
            self.define_metrics()
        metrics_list = self.metrics_list

        return_list = []
        for name, bounds in zip(metrics_list.keys(), metrics_list.values()):
            return_list.append((name, bounds))
        return return_list

    def define_parameters(self, plasiticty_connections=['ee', 'ie'], **kwargs):
        """
        Only ee or ie plasticity connection supported for now.
        """
        parameters = ['alphapre', 'alphapost', 'aplus', 'aminus', 'tauplusstdp', 'tauminusstdp']
        # Following Vogeles et al. bounds for parameters are -2 and 2.0 expect for the time constants that are fixed between 10ms to 100ms 
        parameter_priors = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0,2.0), (0.001, 0.0001), (0.001, 0.0001)]
        all_parameters = {}
        for pconn in plasiticty_connections:
            for param, prior in zip(parameters,parameter_priors) :
                all_parameters[f'{pconn}_{param}'] = prior
        self.parameters = all_parameters
        self.prior_lower_bound = [p[0] for p in parameter_priors]
        self.prior_lower_bound = self.prior_lower_bound * len(plasiticty_connections)
        self.prior_upper_bound = [p[1] for p in parameter_priors]
        self.prior_upper_bound = self.prior_upper_bound * len(plasiticty_connections)
        self.theta_dim = len(self.parameters.keys())
        self.parameters_defined = True

    # Main entry point of the algorithm
    def __call__(self, **kwargs):
        """
        This method is called when the object is called as a function.
        It defines the parameters and metrics of the run based on the provided keyword arguments.

        Parameters:
        - metrics:list (optional): A list of metrics to be used (subset of possible metrics).
        - metric_value:dict (optional): A dictionary specifying specific metric bounds.
        - data_folder:str (optional) : A path where the simulation results will be stored

        Returns:
        None
        """
        # We first define the parameters and metrics of the run
        self.define_parameters(**kwargs)
        self.define_metrics(**kwargs)
        self.data_handler = SimulationHDF5(**kwargs)
        prior = BoxUniform(low=torch.tensor(self.prior_lower_bound), high=torch.tensor(self.prior_upper_bound))
        self.posterior = MakePosterior(
                    theta_dim=self.theta_dim,
                    prior = prior,
                    num_ensemble=self.num_ensemble,
        )
        # Here we want to apply the principle of fSBI
        # We will apply n-conditions at a time

        metrics_list = self.get_metrics_list()
        current_metrics = []
        k = 0
        for metric in metrics_list:
            k +=1
            logger.info(f"--- Starting Round {k} ---")
            current_metrics.append(metric)
            self.round(metrics=current_metrics)
       




    def _make_unique_samples(self, num_samples=None, prior=None, thetas=None, saved_seeds=[]):
        seeds = []
        if prior is None:
            assert thetas is not None
        else:
            thetas = prior.sample(num_samples)
        new_thetas = []
        for th in thetas:
            str_th = str(th).encode()
            seed = hashlib.md5(str_th).hexdigest()
            if seed not in saved_seeds:
                seeds.append(seed)
                new_thetas.append(th)
        return new_thetas, seeds
    

    def round(self, metrics=None, n_samples=10000):
        """
        This method represents a single round of fSBI. 
        The method receives a list of metrics that should be applied as well as the number of samples that should be simulated in this round.

        Functionality of the method:
        (the metrics array must at least contain one metric)
        It is assumed that a posterior has already been trained on the metrics [n-1] where n is the length of the metrics list provided.
        This means that if only 1 metric is provided, no posterior is trained so far.

        First new samples are created. Samples are created from the posterior if necessary (metrics length >= 2) or from the defined prior 

        """

        round_completed = False

        # We implement repetitive mini-rounds with the same metric
        # We go over mini-rounds as long as the samples from the posterior do not satisfy the defined metrics in 0.95
        # of cases 
        logger.info(f"---- Filtering with metrics {metrics} ----")
        while not round_completed:

            # if only one metric is given this means that we are in the first round,
            # we get samples from our defined prior instead of the posterior
            if not metrics:
                logger.warning("Called a simulation round without specified metriecs, skipping.")
                return None
            if len(metrics) == 1:
                # Extract the prior values to a np array 
                priors = self.parameters.values()
                arr_priors = np.empty((0, 2))
                for prior in priors:
            
                    arr_priors = np.vstack((arr_priors, np.array(prior)))
            
                # Use prior values to uniform sample
                thetas = np.random.uniform(arr_priors[:, 0], arr_priors[:, 1], size=(n_samples, len(arr_priors)))
                thetas, _ = self._make_unique_samples(num_samples=n_samples, thetas=thetas)
            
            else:
                """Here we acually want to call a trained posterior ensamble to sample thetas from it"""
                # Extraxt the prior values from the pretrained posterior
                # We will use metrics [n-1] that need to correspond to the input dimensions of the posterior network
                constrain_metrics = metrics[:-1]
                lower_bounds = []
                upper_bounds = []
                for (m_name, m_bounds) in constrain_metrics:
                    lower_bounds.append(m_bounds[0])
                    upper_bounds.append(m_bounds[1])
                # Next we will draw uniform samples from this distribution
                x0 = np.random.uniform(lower_bounds, upper_bounds, (n_samples, len(lower_bounds)))
                x0 = torch.tensor(x0)
                # ############################### 
                #We need to generate samples from a pre-trained posterior here
                # ###############################
                bounds = {'low' : torch.tensor(self.prior_lower_bound), 'high' : torch.tensor(self.prior_upper_bound)}
                thetas = self.posterior.rsample(x0, bounds).detach().numpy().tolist()

            # We simulate new samples, all samples are stored on disk 
            _ , metric_order = self.simulate(thetas)
            self.metric_order = metric_order

            # Now we get all the sample
            
            thetas, obs = self.data_handler.get_filtered_simulation(metrics,  self.metric_order)
            logger.info(f"--- Training posterior on {thetas.shape[0]} samples---")
            self.train_posterior(thetas=thetas, obs=obs)

            # Lets sample to check model quality
            # Now we use all metrics since we already trained the new posterior 
            constrain_metrics = metrics
            lower_bounds = []
            upper_bounds = []
            for (m_name, m_bounds) in constrain_metrics:
                lower_bounds.append(m_bounds[0])
                upper_bounds.append(m_bounds[1])
            # Next we will draw uniform samples from this distribution
            x0 = np.random.uniform(lower_bounds, upper_bounds, (n_samples, len(lower_bounds)))
            x0 = torch.tensor(x0)
            # ############################### 
            #We need to generate samples from a pre-trained posterior here
            # ###############################
            bounds = {'low' : torch.tensor(self.prior_lower_bound), 'high' : torch.tensor(self.prior_upper_bound)}
            thetas  = self.posterior.rsample(x0, bounds).detach().numpy()
            x0 = x0.detach().numpy()
            _, rem_obs = self.data_handler.get_filtered_simulation(constrain_metrics,  self.metric_order, (x0, thetas))
            threshold = 0.95 * n_samples
          
       
            if len(rem_obs) > threshold:
                round_completed = True
                logger.info(f"---- Completing Round with a value of {threshold} ----")
            logger.info(f'---- Next mini round due to low threshold of {threshold}')
            
        return None
    


    def simulate(self, sim_params:list, meta_params:dict={'plasticity': ['ee', 'ie']}, **kwargs):
        """
        This function wrapps the simulation procedure.
        The function gets n parameter sets in a list and runs simulation in parallel.
        The total number of parameters run in parallel are controlled by the num_workers attribute of the class
        Simulation are stored by the instance attribute data_handler
        The function returns None
        """
        metrics = Metrics(broad=True)
        categories = meta_params.get('plasticity')
        assert categories, 'Plasticity categories must be specified!'

        logger.info(f"---- Starting stimulation (simulations: {len(sim_params)}   threads: {self.num_workers}) ----")
      
        args = [{'parameters' : plasticity_parameters(categories, params)} for params in sim_params]
        import multiprocessing
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            """ Simulations (Asynchronous) """
            result_iter = pool.imap(run_simulation, args)
            """ Get results as they are ready """
            output =  []
            i = 0
            for result in result_iter:
                i += 1
                logger.info(f"----- Retrieving raw simulation result {i} / {len(sim_params)} -----")
                m_values, metrics_list = metrics.calculate_metrics(result, return_type='list')
                output.append(m_values)
        output = np.array(output)
        logger.info("---- Saving simulations ----")
        self.data_handler.save_simulations(parameter_set=args, output_set=output)
        return output, metrics_list
       
    def train_posterior(self, thetas, obs, **kwargs):
        """
        This function fits the posterior of the model
        """
        ###################################################
        ### 
        ###
        ### Here we need to implement the posterior fitting
        ###
        ###
        ###################################################
        thetas = torch.tensor(thetas)
        obs = torch.tensor(obs)
      
        self.posterior.get_ensemble_posterior(thetas,
                                         obs,
                                         save_path="./data/posterior_model.pkl",
                                         **kwargs)
        