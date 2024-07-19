import socket
HOST = socket.gethostname()
import argparse
import random
from pathlib import Path
import logging
import brian2 as b2
import hashlib
from brian2.units import *
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from network_analysis import network_run
import json

logging.basicConfig(level=logging.INFO,
                        format=f"run_simulation {HOST}(%(asctime)s) - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("CALLED")
def network_analysis(parameters, working_dir):
    results = network_run(parameters, working_dir)
def kernel_plot(sim_parameters, working_dir):
    """This function is creating the kernel plot of the plasticity rule

    Args:
        parameters (list): list of plasticity parameters (length 12 for ee and ie plasiticy)
        working_dir (str): working directory in which the results will be saved
    """
    # We create the plot for both plasticitz rules
    for i in range(2):
        alpha_pre = sim_parameters[i*6]
        alpha_post =  sim_parameters[i*6+1]
        Aplus = sim_parameters[i*6+2]
        Aminus = -1.0
        tauplus_stdp = sim_parameters[i*6+3] * second
        tauminus_stdp = sim_parameters[i*6+4] * second
        factor = sim_parameters[i*6+5]
        gmax = 100
        lr = 1e-2
        synapse_model = '''
                        w : 1
                        dtrace_pre_plus/dt = -trace_pre_plus / tauplus_stdp : 1 (clock-driven)
                        dtrace_pre_minus/dt = -trace_pre_minus / tauminus_stdp : 1 (clock-driven)
                        dtrace_post_plus/dt = -trace_post_plus / tauplus_stdp : 1 (clock-driven)
                        dtrace_post_minus/dt = -trace_post_minus / tauminus_stdp : 1 (clock-driven)
                        '''
        on_pre = '''
         trace_pre_plus += 1.0
         trace_pre_minus += 1.0
         w = clip(w + lr  * (alpha_pre + factor*Aplus * trace_post_plus + factor*Aminus * trace_post_minus), 0, gmax)
         '''
        on_post = '''
                trace_post_plus += 1
                trace_post_minus += 1
                w = clip(w + lr * (alpha_post + Aplus * trace_pre_plus + Aminus * trace_pre_minus), 0, gmax)
                '''
        firing_rates = [2,3,4,5,8, 10,12, 15, 20, 25,30,35] * Hz

        delta_t = np.linspace(start=-12, stop=12, num=40) * ms
        # Store the final weights for each firing rate and delta_t
        final_weights = {}
        for dttime in tqdm(delta_t, 'Calculating delta t values'):
            final_weights[float(dttime/ms)] = {}
            for rate in firing_rates:
                duration = 5000 * ms
                # Create the neurons
                G = b2.NeuronGroup(2, 'dv/dt = -v / (10*ms) : 1', threshold='v>1', reset='v=0', method='exact')
                G.v = 0

                # Create the synapse with the STDP rule
                S = b2.Synapses(G, G, model=synapse_model, on_pre=on_pre, on_post=on_post,  method='exact')
                S.connect(i=0, j=1)
                S.w = 5.0

                # Define the spike times for pre and post neurons
                pre_spike_times = arange(20*ms, duration + 20*ms, 1/rate)
                post_spike_times = pre_spike_times + dttime

                # Inject spikes using separate spike generators
                spike_gen_pre = b2.SpikeGeneratorGroup(1, [0]*len(pre_spike_times), pre_spike_times)
                spike_gen_post = b2.SpikeGeneratorGroup(1, [0]*len(post_spike_times), post_spike_times)

                # Connect the spike generators to the neuron group
                G_in_pre = b2.Synapses(spike_gen_pre, G, on_pre='v += 1.1', method='exact')
                G_in_pre.connect(i=0, j=0)

                G_in_post = b2.Synapses(spike_gen_post, G, on_pre='v += 1.1',  method='exact')
                G_in_post.connect(i=0, j=1)

                # Create monitors to record spikes and weights
                spikemon_pre = b2.SpikeMonitor(spike_gen_pre)
                spikemon_post = b2.SpikeMonitor(spike_gen_post)
                statemon = b2.StateMonitor(S, ['w'], record=[0])

                # Create a network and run the simulation
                net = b2.Network(b2.collect())
                net.add(G, S, spike_gen_pre, spike_gen_post, G_in_pre, G_in_post, spikemon_pre, spikemon_post, statemon)
                net.run(duration)

                # Record the final weight
                final_weights[float(dttime/ms)][float(rate/Hz)] =  float(5.0-S.w[0])
        # we save the result as a json file
        import json
        name = 'ee' if i == 0 else 'ie'
        with open(working_dir / (name + ".json"), 'w') as f:
            json.dump(final_weights, f)
        # Convert data to numpy array
        d = []
        for k in final_weights.keys():
            v = final_weights[k]
            for o, j in v.items():
                d.append((float(k), float(o), float(j)))
        data = np.array(d)
        delta_t = data[:, 0]
        rate = data[:, 1]
        value = data[:, 2]
        unique_rates = np.unique(rate)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the individual lines and store their data
        plot_data = {}
        # Normalize the rates for the colormap
        # Set custom range for the colormap
        custom_min_value = unique_rates.min() -5 # Or set your custom minimum value
        custom_max_value = unique_rates.max() +5 # Or set your custom maximum value
        norm = Normalize(vmin=custom_min_value, vmax=custom_max_value)
        cmap =  cm.Reds
        for ur in unique_rates:
            dt = delta_t[rate == ur]
            val = value[rate == ur]
            dtplus = dt[dt > 0]
            dtminus = dt[dt <= 0]
            valplus = val[dt > 0]
            valminus = val[dt <= 0]
            plot_data[ur] = (dt, val)
            color = cmap(norm(ur))
            ax.plot(dtplus, valplus, label=f'Rate {ur} Hz',color=color)
            ax.plot(dtminus, valminus, label=f'Rate {ur} Hz',color=color)
        # Add labels and title
        ax.set_xlabel('Delta t (ms)')
        ax.set_ylabel('Weight Change')
        # Create and add a continuous colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax, label='Firing Rate (Hz)')
        # Display the plot
        # Display the plot
        ax.grid(True)
        ax.set_xlim([-15, 15])
        typ = "ee" if i == 0 else "ie"
        plt.title(f"Plasticity Kernel {typ}")
        plt.savefig(str(working_dir / (name + ".png")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some parameters.")
     # Add arguments
    parser.add_argument("--working_dir", type=str)
    parser.add_argument('parameters', metavar='simulation_parameters', type=str,
                                help='list of simulation parameters')
    
    # Parse the arguments
    args = parser.parse_args()
    sim_parameters = args.parameters
    sim_parameters = sim_parameters.split()

    sim_parameters = [float(s) for s in sim_parameters[1:]]
    analysis_dir = Path(args.working_dir) / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting analysis for {}".format(sim_parameters))
    h = hashlib.sha3_256()
    h.update(str.encode(str(random.random())))
    run_id = str(h.hexdigest())
    plot_dir = analysis_dir  / run_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    ############################
    logger.info("Running network")
    valid = network_analysis(sim_parameters, plot_dir)
    logger.info(valid)
    if valid == False:
        try:
            logger.info("UNVALID RUN")
            plot_dir.rmdir()
        except Exception:
            pass
        finally:
            logger.info("Run results not valid - terminating this run ...")
            exit(0)
    else: 
        logger.info("VALID RUN")
    ############################
    logger.info("Creating kernel plot")
    kernel_plot(sim_parameters, plot_dir)
    ###########################
    logger.info("Creating parameter metafile")
    meta_file = plot_dir / "parameters.json"
    meta_dict =  {'ee_alpha_pre' : sim_parameters[0],
                  'ee_alpha_post' : sim_parameters[1],
                  'ee_aplus' : sim_parameters[2],
                  'ee_aminus' : -1.0,
                  'ee_tplus' : sim_parameters[3],
                  'ee_tminus' : sim_parameters[4],
                  'ee_factor' : sim_parameters[5],
                  'ie_alpha_pre' : sim_parameters[6],
                  'ie_alpha_post' : sim_parameters[7],
                  'ie_aplus' : sim_parameters[8],
                  'ie_aminus' : -1.0,
                  'ie_tplus' : sim_parameters[9],
                  'ie_tminus' : sim_parameters[10],
                  'ie_factor' : sim_parameters[11],
                  }
    with open(meta_file, "w") as f:
        json.dump(meta_dict, f)