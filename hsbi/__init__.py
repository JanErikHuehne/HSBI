from concurrent import futures 
from .simulator.network import Rate_STDP_Network, plasticity_parameters, run_simulation
from .density_estimator import MakePosterior
from .metrics import Metrics
import numpy as np 
import logging
import hashlib
import sys
logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

class HSBI: 
    def __init__(self, num_workers=2, ):
        self.num_workers = num_workers
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
                       'f_w-blow' : (0, 0.1),
                       'w_creep' : (0.0, 0.05),
                       'wmean_ee' : (0.0, 0.5),
                       'wmean_ie' : (0.0, 5.0),
                       'mean_fano_t' : (0.5, 2.5),
                       'mean_fano_s' : (0.5, 2.5), 
                       'auto_cov' : (0.0, 0.1),
                       'cv_isi' : (0.7, 1000),
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
        self.parameters_defined = True

    # Main entry point of the algorithm
    def __call__(self, **kwargs):
        """
        This method is called when the object is called as a function.
        It defines the parameters and metrics of the run based on the provided keyword arguments.

        Parameters:
        - metrics:list (optional): A list of metrics to be used (subset of possible metrics).
        - metric_value:dict (optional): A dictionary specifying specific metric bounds.
       

        Returns:
        None
        """
        # We first define the parameters and metrics of the run
        self.define_parameters(**kwargs)
        self.define_metrics(**kwargs)
        # Here we want to apply the principle of fSBI
        # We will apply n-conditions at a time
        print("ROUND 1")
        self.round(metrics=['test'])
    



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
    

    def round(self, metrics=None, n_samples=5):
        """

        """
      
        ###################################################
        ### 
        ###
        ### Here we simulation call needs to be implemented
        ###
        ###
        ###################################################

        # if only one metric is given this means that we are in the first round,
        # we get samples from our defined prior instead of the posterior
        if not metrics:
            logger.warning("Called a simulation round without specified metrics, skipping.")
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

        outputs = self.simulate(thetas)
        return outputs
    


    def simulate(self, sim_params:list, meta_params:dict={'plasticity': ['ee', 'ie']}, **kwargs):
        """
        This function wrapps the simulation procedure.
        The function gets n parameter sets in a list and runs simulation in parallel.
        The total number of parameters run in parallel are controlled by the num_workers attribute of the class
        The function returns a list of calculated summary methods 
        """
        metrics = Metrics(broad=True)
        categories = meta_params.get('plasticity')
        assert categories, 'Plasticity categories must be specified!'

        print("SIMULATING")
      
        args = [{'parameters' : plasticity_parameters(categories, params)} for params in sim_params]
        import multiprocessing
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            """ Simulations (Asynchronous) """
            result_iter = pool.imap(run_simulation, args)
            """ Get results as they are ready """
            output =  []
            for result in result_iter:
                logger.debug("Retrieving result ...")
                output.append(metrics.calculate_metrics(result, return_type='list'))
        output = np.array(output)
        print(output.shape)
        return output
       
    def fit_posterior(self, thetas, obs, num_ensemble=None, **kwargs):
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

        """
        posterior = MakePosterior(
                    theta_dim=self.theta_dim,
                    low_lim=self.low_prior_bound,
                    up_lim=self.up_prior_bound,
                    num_ensemble=num_ensemble if not None else 10,
        )
        posterior.get_ensemble_posterior(thetas,
                                         obs,
                                         **kwargs)
        """