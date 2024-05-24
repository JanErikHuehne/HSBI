import argparse
import brian2 as b2
from brian2.units import *
import logging
import os 
from pathlib import Path
import numpy as np 
import h5py
from collections.abc import Iterable
import time 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

b2.prefs.codegen.target = "numpy"

def time_function(func): 
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        run_id = kwargs['run_id']
        import socket
        host = socket.gethostname()
        logger.info(f"{host}:{func.__name__}({run_id}) executed in {elapsed:.6f} seconds")
        return result
    return wrapper

def save_results(file_path, results):
    with h5py.File(file_path, 'w') as h:
        for k, v in results.items():
            if isinstance(v, dict):
                group = h.create_group(k)
                for key, val in v.items():
                    if isinstance(val, np.ndarray):
                        group.create_dataset(key, data=val)
                    elif isinstance(val, str):
                        # Store strings as fixed-size ASCII to ensure compatibility
                        dt = h5py.string_dtype('ascii')
                        group.create_dataset(key, data=np.string_(val), dtype=dt)
                    else:
                        # Convert other types to strings or appropriate handling
                        group.create_dataset(key, data=str(val))
            else:
                if isinstance(v, np.ndarray):
                    h.create_dataset(k, data=v)
                elif isinstance(v, str):
                    dt = h5py.string_dtype('ascii')
                    h.create_dataset(k, data=np.string_(v), dtype=dt)
                else:
                    h.create_dataset(k, data=str(v))


def metrics(result):
    bin_size_big = 0.1
    bin_size_medium = 0.01
    bin_size_small = 0.001
    window_view_auto_cov = 0.5

    def extract_neuron_spikes(self, spike_times, neuron_ids):
            neuron_spikes = {}
            for i in range(len(neuron_ids)):
                neuron_id = neuron_ids[i]
                if not neuron_spikes.get(neuron_id):
                    neuron_spikes[neuron_id] = []
                neuron_spikes[neuron_id].append(spike_times[i])
            return neuron_spikes
    
    def rate_e(sim_data):
        ####################################
        # This method computes the global firing rate of the inhibitory neuron population
        # We access the spike data of the inibitory neurons
        ####################################

        spike_data = sim_data['spikes_pe']
        num_neurons = spike_data['num_neurons']

        # spike times
        spike_times = spike_data['times']

        total_num_of_spikes = len(spike_times)

        # firing rate
        rate = total_num_of_spikes / (num_neurons * sim_data['runtime'])
        return rate
    
    def rate_i(sim_data):
        ####################################
        # This method computes the global firing rate of the inhibitory neuron population
        # We access the spike data of the inibitory neurons
        ####################################

        spike_data = sim_data['spikes_pi']
        num_neurons = spike_data['num_neurons']

        # spike times
        spike_times = spike_data['times']

        total_num_of_spikes = len(spike_times)

        # firing rate
        rate = total_num_of_spikes / (num_neurons * sim_data['runtime'])
        return rate
    
    def weef(sim_data):
        """final mean EE weight"""
        w_trace = sim_data['weights_ee']['weights']
        return np.mean(w_trace[:, -1])
         

    def wief(sim_data):
        """final mean IE weight"""
        w_trace = sim_data['weights_ie']['weights']
        return np.mean(w_trace[:, -1])
    
    result['rate_e'] = rate_e(result)
    result['rate_i'] = rate_i(result)
    result['wmean_ee'] = weef(result)
    result['wmean_ie'] = wief(result)
    del result['weights_ee']
    del result['weights_ie']
    return result
@time_function
def simulation(sim_params, run_id):
    b2.start_scope()
    """Shared network parameters"""
    NE = 200
    NI = NE / 4
    input_num = 100
    input_freq = 30 # Hz
    sim_time = 25
    gmax = 100.0
    lr = 1e-2 
    Aminus = 1.0
    epsilon = 0.1
    gl = 10 * nS
    er = -80 * mV
    el = - 60 * mV
    tau_gaba = 10.0 *ms
    tau_ampa = 5.0 * ms
    vt = -50 * mV
    bgcurrent = 200 * pA
        
    memc = 200 * pfarad
    eqs_neurons='''
                dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
                dg_ampa/dt = -g_ampa/tau_ampa : siemens
                dg_gaba/dt = -g_gaba/tau_gaba : siemens
                '''
    neurons = b2.NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                             reset='v=el', refractory=5*ms, method='euler')
    Pe = neurons[:NE]
    Pi = neurons[NE:]

    # EE plasticity 
    ee_alpha_pre = sim_params[0]
    ee_alpha_post = sim_params[1]
    ee_Aplus = sim_params[2]
    ee_tauplus_stdp = sim_params[3] * ms
    ee_tauminus_stdp = sim_params[4] * ms
    ee_Aminus = -1.0
  
    synapse_model ='''
                w : 1
                dee_trace_pre_plus/dt = -ee_trace_pre_plus / ee_tauplus_stdp : 1 (event-driven)
                dee_trace_pre_minus/dt = -ee_trace_pre_minus / ee_tauminus_stdp : 1 (event-driven)
                dee_trace_post_plus/dt = -ee_trace_post_plus / ee_tauplus_stdp : 1 (event-driven)
                dee_trace_post_minus/dt = -ee_trace_post_minus / ee_tauminus_stdp : 1 (event-driven)
    '''
    con_ee = b2.Synapses(Pe, Pe, model=synapse_model,
                                on_pre='''
                                        g_ampa += w*nS
                                        ee_trace_pre_plus += 1.0
                                        ee_trace_pre_minus += 1.0
                                        w = clip(w + lr * (ee_alpha_pre + ee_Aplus*ee_trace_post_plus + ee_Aminus * ee_trace_post_minus), 0, gmax)
                                        ''',
                                on_post='''
                                        ee_trace_post_plus += 1
                                        ee_trace_post_minus += 1
                                        w = clip(w + lr * (ee_alpha_post + ee_Aplus*ee_trace_pre_plus + ee_Aminus * ee_trace_pre_minus), 0, gmax)
                                        '''
                                )
    con_ee.connect(p=epsilon)
    con_ee.w = 0.1
    # EI Plasticity
    con_ei = b2.Synapses(Pe, Pi, on_pre="g_ampa += 0.1*nS")
    con_ei.connect(p=epsilon)
   
    #  II Plasticity
    con_ii = b2.Synapses(Pi,Pi, on_pre="g_gaba += 1*nS")
    con_ii.connect(p=epsilon)
    # IE Plasiticty 
    ie_alpha_pre = sim_params[5]
    ie_alpha_post = sim_params[6]
    ie_Aplus =  sim_params[7]
    ie_tauplus_stdp = sim_params[8] * ms
    ie_tauminus_stdp = sim_params[9] * ms
   
    ie_Aminus = -1.0
    synapse_model ='''
                w : 1
                die_trace_pre_plus/dt = -ie_trace_pre_plus / ie_tauplus_stdp : 1 (event-driven)
                die_trace_pre_minus/dt = -ie_trace_pre_minus / ie_tauminus_stdp : 1 (event-driven)
                die_trace_post_plus/dt = -ie_trace_post_plus / ie_tauplus_stdp : 1 (event-driven)
                die_trace_post_minus/dt = -ie_trace_post_minus / ie_tauminus_stdp : 1 (event-driven)
                '''
    con_ie = b2.Synapses(Pi, Pe, model=synapse_model,
                                on_pre='''
                                        g_gaba += w*nS
                                        ie_trace_pre_plus += 1.0
                                        ie_trace_pre_minus += 1.0
                                        w = clip(w + lr * (ie_alpha_pre + ie_Aplus * ie_trace_post_plus + ie_Aminus * ie_trace_post_minus), 0, gmax)
                                        ''',
                                on_post='''
                                        ie_trace_post_plus += 1
                                        ie_trace_post_minus += 1
                                        w = clip(w + lr * (ie_alpha_post + ie_Aplus * ie_trace_pre_plus + ie_Aminus * ie_trace_pre_minus), 0, gmax)
                                        '''
                        )
    con_ie.connect(p=epsilon)
    con_ie.w = 1.0
    neurons.v = 0
    P = b2.PoissonGroup(input_num, input_freq*Hz)
    # We only input to the exitatory population 
    S = b2.Synapses(P, Pe, on_pre='g_ampa += 0.3*nS').connect(p=0.3)
    b2.run(sim_time * second)
    # Define monitors
    MPe = b2.SpikeMonitor(Pe)
    MPi = b2.SpikeMonitor(Pi)
    W_IE = b2.StateMonitor(con_ie, 'w', record=True)
    W_EE = b2.StateMonitor(con_ee, 'w', record=True)
    b2.run(5 * second)
    # Result retrieval
    spikes = {}
    times = MPe.t
    neurons_ids = MPe.i
    spikes['Pe'] = {'type' : 'exictatory',
                        'num_neurons' : NE,
                        'times' : np.array(times).copy(),
                        'neurons' : np.array(neurons_ids).copy()}
    times = MPi.t
    neurons_ids = MPi.i
    spikes['Pi'] = {'type' : 'inhibitory',
                    'num_neurons' : NI,
                    'times' :  np.array(times).copy(),
                    'neurons' :  np.array(neurons_ids).copy()}
    weights = {}
    weights['ie'] = {
                                'from' :'Pi',
                                'to' : 'Pe',
                                'weights' : np.array(W_IE.w).copy()
                    }
    weights['ee'] = {
                                'from' :'Pe',
                                'to' : 'Pe',
                                'weights' : np.array(W_EE.w).copy()
                    }
    return  metrics({'run_parameters': sim_params,
         'spikes_pe': spikes['Pe'],
                'spikes_pi' : spikes['Pi'],
                'weights_ie' : weights['ie'],
                'weights_ee' : weights['ee'],
                 'runtime' : sim_time,
                 't_start' : 0,#pre_simtime,
                 't_end' : sim_time,#pre_simtime+simtime,
                 'dt' : float(b2.defaultclock.dt)})
        
if __name__ == "__main__":
        # Create the argument parser
        parser = argparse.ArgumentParser(description="Process some parameters.")

        # Add arguments
        parser.add_argument("--working_dir", type=str)
        parser.add_argument('parameters', metavar='simulation_parameters', type=str,
                                help='list of simulation parameters')
        
        # Parse the arguments
        args = parser.parse_args()
        sim_parameters = args.parameters
        sim_parameters = sim_parameters.split()
        run_id = int(sim_parameters[0])
        
        sim_parameters = [float(s) for s in sim_parameters[1:]]
        result = simulation(sim_parameters, run_id=run_id)
        logger.error(f"Result {result['rate_e']} {result['rate_i']} for {result['run_parameters'][-2:]}")
        
        # now we save the raw simulation results 

        temp_sim_runs = Path(args.working_dir) / "raw_results"
        os.makedirs(temp_sim_runs, exist_ok=True)

        import hashlib 
        import random
        ex = True
        while ex: 
                h = hashlib.sha3_256()
                h.update(str.encode(str(random.random())))
                run_id = str(h.hexdigest()) + ".hdf5"
                file = temp_sim_runs / run_id
                if not file.exists():
                        ex = False
        save_results(file, result)
       


  


   
