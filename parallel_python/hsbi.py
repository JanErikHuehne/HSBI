import torch 
import logging 
import hashlib
import numpy as np 
from sbi.utils import BoxUniform
from density_estimator import *
from pathlib import Path
import argparse
import sys
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from utils import collect_simulations
import random
N_SIZE = 20000
logging.basicConfig(level=logging.INFO,
                        format='MAIN tuwzc1n-cortex(%(asctime)s) - %(levelname)s - %(message)s', datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HSBI:
    
    def __init__(self, num_ensamble=5, **kwargs):
        self.num_ensamble = num_ensamble
        self.define_metrics(**kwargs)
        self.define_parameters(**kwargs)
    
    def train_posterior(self, thetas, obs, save_path, **kwargs):
        """ Train a posterior ensemble on data 

        Args:
            thetas (torch.tensor): a tensor of plasticity parameters 
            obs (torch.tensor): a tensor of observed metrics
            save_path (str or pathlib.path): savepath of posterior object (pkl file)
          
        """
        ###################################################
        ### 
        ###
        ### Here we need to implement the posterior fitting
        ###
        ###
        ###################################################
       
        prior = BoxUniform(low=torch.tensor(self.prior_lower_bound), high=torch.tensor(self.prior_upper_bound))
        self.posterior = MakePosterior(
                    theta_dim=4,
                    prior = prior,
                    num_ensemble=self.num_ensamble,
        )
        self.posterior.get_ensemble_posterior(thetas,
                                         obs,
                                         save_path=save_path,
                                         **kwargs)
        
    def define_parameters(self, plasiticty_connections=['ee', 'ie'], **kwargs):
        """
        Only ee or ie plasticity connection supported for now.
        """
        parameters = ['alphapre', 'alphapost', 'aplus', 'tauplusstdp', 'tauminusstdp']
        # Following Vogeles et al. bounds for parameters are -2 and 2.0 expect for the time constants that are fixed between 10ms to 100ms 
        # tau values should be set to 5ms - 30ms
        parameter_priors = [(-1.0, 1.0), (-1.0, 1.0), (0.2,5.0), (0.005, 0.030), (0.005, 0.030)]
        all_parameters = {}
        for pconn in plasiticty_connections:
            for param, prior in zip(parameters,parameter_priors) :
                all_parameters[f'{pconn}_{param}'] = prior
        # We add the terms for for the factors 
        all_parameters[f'factor_ee'] = [(-2, -0.5), (0.5,2)]
        all_parameters[f'factor_ie'] = [(-2, -0.5), (0.5,2)]
        self.parameters = all_parameters
        self.prior_lower_bound = [p[0] for p in parameter_priors]
        self.prior_lower_bound = self.prior_lower_bound * len(plasiticty_connections)
        self.prior_upper_bound = [p[1] for p in parameter_priors]
        self.prior_upper_bound = self.prior_upper_bound * len(plasiticty_connections)
        self.theta_dim = len(self.parameters.keys())
        self.parameters_defined = True

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
       
        self.metrics_list = {'rate_e' : (1,50),
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
        # We set the metrics to the subset of metrics if they have been specified 
        if metrics is not None:
          
            logger.info(f"Found metrics")
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


def make_unique_valid_samples(num_samples=None, prior=None, thetas=None, saved_seeds=[], get_info=True):
    """This method is used to create unique samples that comply with the area under the curve constrain of the plasticity kernel of (-1, 1)
    """
    seeds = []
    if prior is None:
        assert thetas is not None
    else:
        thetas = prior.sample(num_samples)
    new_thetas = []
    bpp = 0
    bnn = 0
    bpn = 0
    bnp = 0
    for th in thetas: 
        str_th = str(th).encode()
        seed = hashlib.md5(str_th).hexdigest()
        if seed not in saved_seeds:
            # We check for the B value here to be in the range of -1 to 1
            categories = len(th) / 6
            use = True
            Bs = []
           
            # Iterate over each plasticity category containing 5 plasticity parameters
            # Calculate the B value (Aminus = -1.0)
            for i in range(int(categories)):
                Aplus = th[i*6 + 2]
                tauplus = th[i*6 +3]
                tauminus = th[i*6 +4]
                factor = th[i*6 +5]
                B = (1+factor) * (Aplus * tauplus - tauminus)
                Bs.append(B)
                if Aplus < 0.2 or Aplus > 5.0:
                    #logger.info("Wrong Aplus {}".format(Aplus))
                    use = False
            if use: 
                seeds.append(seed)
                #th = np.append(th, Bs)
                if Bs[0] > 0 and Bs[1] > 0:
                    bpp += 1
                elif Bs[0] > 0 and Bs[1] < 0:
                    bpn += 1
                elif Bs[0] < 0 and Bs[1] > 0:
                    bnp += 1
                else:
                    bnn += 1
                new_thetas.append(th)
            else:
                if get_info:
                    logger.info("Not a valid simulation, not using simulation")
    if get_info:
        logger.info(f"Using {len(new_thetas)} / {len(thetas)} with (b++,b+-,b-+,b--) : ({bpp},{bpn},{bnp},{bnn})")
    return new_thetas, seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', action='store_true', help="Sample simulation parameters")
    group.add_argument("--train",action='store_true', help="Train a new posterior")
    parser.add_argument("--working_dir", type=str)
    parser.add_argument('metrics', metavar='metrics', type=str,
                                help='list of metric categories to be applied', default="")
    args = parser.parse_args()
    # Train or sample mode 
    train_mode = args.train 

    metrics = args.metrics
    metrics = metrics.split()
    working_directory = args.working_dir
    simulation_results = Path(working_directory) / "simulations.hdf5"
    parameter_output = (Path(working_directory) / Path("parameters.txt")).resolve()
    logger.info(parameter_output)
    posterior_dir = Path(working_directory) / "posteriors"
    posterior_dir.mkdir(exist_ok=True)
    # Code for training posterior
    hsbi = HSBI()
    if train_mode:
        if not metrics:
            logger.info(f"No need to train a posterior without samples - Skipping")
            sys.exit(0)
        ###
        # Case: Train with filtered metrics 
        ###
        else: 
            # We collect all the simulations
            posterior_file = "_".join(metrics) + ".pkl"
            save_path = posterior_dir / posterior_file
            logger.info("Collecting simulations")
            thetas, obs = collect_simulations(metrics=metrics, metrics_bounds=hsbi.metrics_list, h5_path=simulation_results)

            logger.info(f"Training on {len(thetas)} samples of shape {thetas.shape}")
            hsbi.train_posterior(thetas=thetas, obs=obs, save_path=save_path)
    else: 

        ###
        # Case: Sample, no metrics provided --> Sample from inital prior
        ###
        
        if not metrics:
            logger.info("Creating new samples from inital prior")
            found_thetas = False
            total_thetas = None
            while not found_thetas:
                priors = hsbi.parameters.values()
                arr_priors = np.empty((0, 2))
                for prior in priors:  
                    if isinstance(prior, list):
                        arr_priors = np.vstack((arr_priors, np.array(random.choice(prior))))
                    else:
                        arr_priors = np.vstack((arr_priors, np.array(prior)))
                # Idea first sample B values
                bs = np.random.uniform(-0.05, 0.05, size=(N_SIZE, 2))
                # We use tauminus as the value to fit to the B values 
            
                thetas = np.random.uniform(arr_priors[:, 0], arr_priors[:, 1], size=(N_SIZE, len(arr_priors)))
                aplus1 = (thetas[:,4] + (bs[:,0] / (1+thetas[:,5]) )) / thetas[:,3] 
                aplus2 = (thetas[:,10] + (bs[:,1] / (1+thetas[:,11]) )) / thetas[:,9] 
               
                thetas[:,2] = aplus1
                thetas[:,8] = aplus2
                thetas, _ = make_unique_valid_samples(num_samples=N_SIZE, thetas=thetas)
                thetas = np.array(thetas)
                if len(thetas) > 0:
                    if total_thetas is None:
                        total_thetas = thetas
                    else: 
                        total_thetas = np.concatenate((total_thetas, thetas))
                
                if total_thetas is not None and len(total_thetas) >= N_SIZE:
                    found_thetas = True  
                    total_thetas, _ = make_unique_valid_samples(num_samples= len(total_thetas), thetas=total_thetas, get_info=True) 
                    total_thetas = np.array(total_thetas)
                    logger.info(f"Created {len(total_thetas)} new simulations")
                    with open(parameter_output, "w") as f:
                        i = 0
                        lines = []
                        for th in total_thetas:
                            i += 1
                            string_parm_list = ['{:.10f}'.format(x) for x in th]
                            parm_line = " ".join(string_parm_list)
                            parm_line = f"{i} " + parm_line
                            parm_line += "\n"
                            lines.append(parm_line)
                        f.writelines(lines)
                else:
                    if total_thetas is not None:
                        if len(total_thetas) % 10 == 0:
                            logger.info(f"Not enough samples ({len(total_thetas)}) created yet ...")

        ###
        # Case: Sample. metrics provided --> Sample from pre-trained posterior
        ###
        else:
            lower_bounds = []
            upper_bounds = []
            # Get metrics constrains
            for m in metrics: 
                bounds = hsbi.metrics_list[m]
                lower_bounds.append(bounds[0])
                upper_bounds.append(bounds[1])
         
            x0 = np.random.uniform(lower_bounds, upper_bounds, (N_SIZE, len(lower_bounds)))
            x0 = torch.tensor(x0)
            bounds = {'low' : torch.tensor(hsbi.prior_lower_bound), 'high' : torch.tensor(hsbi.prior_upper_bound)}
            posterior_file = "_".join(metrics) + ".pkl"
            save_path = posterior_dir / posterior_file
            assert save_path.exists(), "Could not read posterior file"
            with open(save_path, "rb") as f:
                posterior = pickle.load(f)
            done = False
            while not done: 
                try: 
                    thetas = posterior.rsample(x0, bounds).detach().numpy().tolist()
                    done = True
                except Exception as e: 
                    logger.info(e.with_traceback())
                    logger.warning("Sampling from posterior was not successful - trying again")
                    done = False
            thetas, _ = make_unique_valid_samples(num_samples=N_SIZE, thetas = thetas)
            with open(parameter_output, "w") as f:
                i = 0
                for th in thetas:
                    string_parm_list = ['{:.10f}'.format(x) for x in th]
                    parm_line = " ".join(string_parm_list)
                    full_line = f"{i} " + parm_line
                    f.write(full_line)
                    f.write("\n")
                    i += 1
        sys.exit(0)
   