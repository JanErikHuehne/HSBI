import socket
HOST = socket.gethostname()
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
from scipy.signal import correlate
logging.basicConfig(level=logging.INFO,
                        format=f"run_simulation {HOST}(%(asctime)s) - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
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
    gmax = 20.0

    def extract_neuron_spikes(spike_times, neuron_ids):
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
         
    def w_creep(sim_data):
        """ 
        creep of the weights. Change of the mean weight between start and finish as percentage,
        taken as the maximum compared between different plasticity connections
        """
        w_trace_ee = sim_data['weights_ee']['weights']
        start_w_ee = np.mean(w_trace_ee[:,0])
        end_w_ee = np.mean(w_trace_ee[:, -1])
        if start_w_ee + end_w_ee > 0.1:
             creep_ww = np.abs(2* (end_w_ee - start_w_ee) / (end_w_ee + start_w_ee))
        else:
             crep_ww = 0.0
        w_trace_ie = sim_data['weights_ie']['weights']
        start_w_ie = np.mean(w_trace_ie[:,0])
        end_w_ie = np.mean(w_trace_ie[:, -1])
        if start_w_ie + end_w_ie > 0.1:
             creep_ie = np.abs(2* (end_w_ie - start_w_ie) / (end_w_ie + start_w_ie))
        else:
             crep_ie = 0.0
        return max(creep_ww, crep_ie)
    
    def w_blow(sim_data):
        """Percentage of weights that did blow up to a maximum value"""
        f_blow = 0.0
        for conn in ['weights_ee', "weights_ie"]:
            weights = sim_data[conn]['weights']
            # we get the maximum weight value of each trace
            max_w = np.max(weights, axis=1)
            min_w = np.max(weights, axis=1)
            blow = np.sum(max_w > gmax)
            vanish = np.sum(min_w < 0.001)
            f_blow += (blow + vanish) / (len(max_w))
        return f_blow / 2
    
    def cv_isi(sim_data):
        """coefficient of variation of the interspike intervals of exitatory neuron population"""
        spikes = sim_data['spikes_pi']
        t_spikes = spikes['times']
        neuron_ids = spikes['neurons']
        grouped_spikes = extract_neuron_spikes(t_spikes, neuron_ids)
        var_isi_val = []
        for key,val in zip(grouped_spikes.keys(), grouped_spikes.values()):
            isi = np.std(np.diff(val)) / np.mean(np.diff(val))
            var_isi_val.append(isi)
        return np.mean(var_isi_val)
    
    def wief(sim_data):
        """final mean IE weight"""
        w_trace = sim_data['weights_ie']['weights']
        return np.mean(w_trace[:, -1])
    
    def std_fr_s(sim_data):
        """
        Standard deviation of the firing rate over spacial domain  
        """

     
        grouped_spikes_e =  extract_neuron_spikes(sim_data['spikes_pe']['times'], sim_data['spikes_pe']['neurons'])
        grouped_spikes_i =  extract_neuron_spikes(sim_data['spikes_pi']['times'], sim_data['spikes_pi']['neurons'])
       
        all_rates = []
        # exitatory
        for spike_train in grouped_spikes_e.values():
          all_rates.append(len(spike_train) / sim_data['runtime'])

        # inhibitory
        for spike_train in grouped_spikes_i.values():
            all_rates.append(len(spike_train) / sim_data['runtime'])
        
        return np.std(np.array(all_rates))

    def std_fr(sim_data):
        """standard deviation of the firing rate
            FR are computed over successive 1ms time windows, on which the standard deviation was computed
        """
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'],bin_size_medium)
        grouped_spikes_e =  extract_neuron_spikes(sim_data['spikes_pe']['times'], sim_data['spikes_pe']['neurons'])
        grouped_spikes_i =  extract_neuron_spikes(sim_data['spikes_pi']['times'], sim_data['spikes_pi']['neurons'])
        # exitatory population
        stds = []
        for spike_train in grouped_spikes_e.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            std = np.std(binned_spike_train)
            stds.append(std)

        # inhibitory population
        for spike_train in grouped_spikes_i.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            std = np.std(binned_spike_train)
            stds.append(std)
        return np.mean(stds)

    def auto_cov(sim_data):
        """
        autocov of the spike train
        Spike trains are binned in a time window of 10ms
        autocov for each neuron is computed and normalized between -500 and 500 ms
        Mean area under the curve is computed and averaged over neurons
        """
        grouped_spikes_e = extract_neuron_spikes(sim_data['spikes_pe']['times'], sim_data['spikes_pe']['neurons'])
        grouped_spikes_i = extract_neuron_spikes(sim_data['spikes_pi']['times'], sim_data['spikes_pi']['neurons'])
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'], bin_size_medium)
        binned = np.empty(shape=(0, len(tbins)-1))
        # exitatory
        for spike_train in grouped_spikes_e.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned = np.vstack([binned, binned_spike_train])

        # inhibitory
        for spike_train in grouped_spikes_i.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned = np.vstack([binned, binned_spike_train])
        # Correlate
        lags = int( window_view_auto_cov / bin_size_medium)
        aoc = []
        for x in binned:
            x_corr = correlate(x - x.mean(), x- x.mean(), mode='full')
            # normalize
            x_corr = np.abs(x_corr) / x_corr.max()
            aoc.append(np.mean(x_corr[(x_corr.size//2-(lags)):(x_corr.size//2+(lags+1))]))
        return np.mean(aoc)
    
    def averaged_fano_spatial(sim_data):
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'], bin_size_big)
        grouped_spikes_e =  extract_neuron_spikes(sim_data['spikes_pe']['times'], sim_data['spikes_pe']['neurons'])
        grouped_spikes_i =  extract_neuron_spikes(sim_data['spikes_pi']['times'], sim_data['spikes_pi']['neurons'])
        binned = np.empty(shape=(0, len(tbins)-1))
        
        # exitatory
        for spike_train in grouped_spikes_e.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned = np.vstack([binned, binned_spike_train])

        # inhibitory
        for spike_train in grouped_spikes_i.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned = np.vstack([binned, binned_spike_train])
        
        # compute the fano factor over each time window 
        mean = np.mean(binned, axis=0)
        var = np.var(binned, axis=0)
        fano = var / mean
        return np.mean(fano)
    
    def averaged_fano_time(sim_data):
        """fano factor for each spike train averaged over the population
            Binning the spike-trains over 100ms successive windows 
            Fano Factor computed per neuron, then averaged over the population
        """
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'],bin_size_big)

        grouped_spikes_e =  extract_neuron_spikes(sim_data['spikes_pe']['times'], sim_data['spikes_pe']['neurons'])
        grouped_spikes_i =  extract_neuron_spikes(sim_data['spikes_pi']['times'], sim_data['spikes_pi']['neurons'])
        # exitatory population
        ffs = []
        for spike_train in grouped_spikes_e.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            if np.sum(binned_spike_train) <= 3:
                continue
            mean = np.mean(binned_spike_train)
            var = np.var(binned_spike_train)
            ffs.append(var / mean)
        
        # inhibitory population
        for spike_train in grouped_spikes_i.values():

            binned_spike_train = np.histogram(spike_train, tbins)[0]
            if np.sum(binned_spike_train) <= 3:
                continue

            mean = np.mean(binned_spike_train)
            var = np.var(binned_spike_train)
            ffs.append(var / mean)
        fano = np.mean(ffs) 
        return fano
    
    result['rate_e'] = rate_e(result)
    result['rate_i'] = rate_i(result)
    result['wmean_ee'] = weef(result)
    result['wmean_ie'] = wief(result)
    result['f_w-blow'] = w_blow(result)
    result['cv_isi'] = cv_isi(result)
    result['std_fr'] = std_fr(result)
    result['std_rate_spatial'] = std_fr_s(result)
    result['auto_cov'] = auto_cov(result)
    result['mean_fano_s'] = averaged_fano_spatial(result)
    result['mean_fano_t'] = averaged_fano_time(result)
    # We delete the raw data so its not saved, simulations can be repeated with the given seed
    del result['weights_ee']
    del result['weights_ie']
    del result['spikes_pi']
    del result['spikes_pe']
    return result

@time_function
def simulation(sim_params, run_id, seed=None):
    b2.start_scope()
    # set random seed
    if seed is None:
        seed = np.random.randint(low=1)
    b2.seed(seed)
    """Shared network parameters"""
    NE = 400
    NI = NE / 4
    input_num = 100
    input_freq = 30 # Hz
    sim_time = 25
    gmax = 20.0
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
    con_ee.w = 0.2
    # EI Plasticity
    con_ei = b2.Synapses(Pe, Pi, on_pre="g_ampa += 0.2*nS")
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
                'seed' : seed,
                'spikes_pi' : spikes['Pi'],
                'weights_ie' : weights['ie'],
                'weights_ee' : weights['ee'],
                 'runtime' : 5,
                 't_start' : sim_time,#pre_simtime,
                 't_end' : sim_time + 5,#pre_simtime+simtime,
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
       


  


   
