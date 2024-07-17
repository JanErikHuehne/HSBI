import socket
HOST = socket.gethostname()
import brian2 as b2
from brian2.units import *
import logging
import time 
import numpy as np 

import json
from pathlib import Path
import warnings
warnings.filterwarnings("error")

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
        import socket
        host = socket.gethostname()
        logger.info(f"{host}:{func.__name__}) executed in {elapsed:.6f} seconds")
        return result
    return wrapper


@time_function
def network_run(sim_params, working_directory):
    logger.info(f"Running in {working_directory}")
    b2.start_scope()
    """Shared network parameters"""
    NE = 400
    NI = NE // 4
    input_num = 40
    input_freq = 30 # Hz
    sim_time = 120
    gmax = 20.0
    lr = 1e-2 
    Aminus = 1.0
    epsilon = 0.2
    gl = 10 * nS
    er = -80 * mV
    el = - 60 * mV
    tau_gaba = 10.0 * ms
    tau_ampa = 5.0 * ms
    vt = -50 * mV
    memc = 200 * pfarad
    eqs_neurons = '''
                dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
                dg_ampa/dt = -g_ampa/tau_ampa : siemens
                dg_gaba/dt = -g_gaba/tau_gaba : siemens
                '''
    neurons = b2.NeuronGroup(NE + NI, model=eqs_neurons, threshold='v > vt',
                             reset='v=el', refractory=5*ms, method='euler')
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    # EE plasticity 
    ee_alpha_pre = sim_params[0]
    ee_alpha_post = sim_params[1]
    ee_Aplus = sim_params[2]
    ee_tauplus_stdp = sim_params[3] * ms
    ee_tauminus_stdp = sim_params[4] * ms
    factor_ee = sim_params[5]
    ee_Aminus = -1.0
    synapse_model = '''
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
                                w = clip(w + lr * factor_ee * (ee_alpha_pre + ee_Aplus*ee_trace_post_plus + ee_Aminus * ee_trace_post_minus), 0, gmax)
                                ''',
                         on_post='''
                                ee_trace_post_plus += 1
                                ee_trace_post_minus += 1
                                w = clip(w + lr * (ee_alpha_post + ee_Aplus*ee_trace_pre_plus + ee_Aminus * ee_trace_pre_minus), 0, gmax)
                                '''
                         )
    con_ee.connect(p=epsilon, condition='i != j')
    con_ee.w = 0.2
    # EI Plasticity
    con_ei = b2.Synapses(Pe, Pi, on_pre="g_ampa += 0.2*nS")
    con_ei.connect(p=epsilon, condition='i != j')
    # II Plasticity
    con_ii = b2.Synapses(Pi, Pi, on_pre="g_gaba += 1*nS")
    con_ii.connect(p=epsilon, condition='i != j')
    # IE Plasticity 
    ie_alpha_pre = sim_params[6]
    ie_alpha_post = sim_params[7]
    ie_Aplus = sim_params[8]
    ie_tauplus_stdp = sim_params[9] * ms
    ie_tauminus_stdp = sim_params[10] * ms
    factor_ie = sim_params[11]
    ie_Aminus = -1.0
    synapse_model = '''
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
                                w = clip(w + lr * factor_ie * (ie_alpha_pre + ie_Aplus * ie_trace_post_plus + ie_Aminus * ie_trace_post_minus), 0, gmax)
                                ''',
                         on_post='''
                                ie_trace_post_plus += 1
                                ie_trace_post_minus += 1
                                w = clip(w + lr * (ie_alpha_post + ie_Aplus * ie_trace_pre_plus + ie_Aminus * ie_trace_pre_minus), 0, gmax)
                                '''
                         )
    con_ie.connect(p=epsilon, condition='i != j')
    con_ie.w = 1.0
    neurons.v = 0
    P = b2.PoissonGroup(input_num, input_freq * Hz)
    S = b2.PoissonInput(Pe, N=input_num, target_var="g_ampa", rate=20 * Hz, weight=0.4 * nS)
    results = {}
    try:
        b2.run(10 * second, report='text')
    except RuntimeWarning:
            logger.info("Runtime Warning found. Terminating this run ...")
            return False
    # First Recording Window
    MPe_All = b2.SpikeMonitor(Pe)
    MPi_All = b2.SpikeMonitor(Pi)
    W_IE = b2.StateMonitor(con_ie, 'w', record=True, dt=0.5 * second)
    W_EE = b2.StateMonitor(con_ee, 'w', record=True, dt=0.5 * second)
    b2.run(30 * second, report='text')
    # We get the information of the spike monitors
    results[0] = {
        'start': 10,
        'end': 40,
        'ee_times': np.array(MPe_All.t / second, dtype=float).tolist(),  # Convert times to seconds
        'ee_neuron_ids': np.array(MPe_All.i, dtype=int).tolist(),
        'ie_times': np.array(MPi_All.t / second, dtype=float).tolist(),  # Convert times to seconds
        'ie_neuron_ids': np.array(MPi_All.i, dtype=int).tolist(),
        'w_ee': np.array(W_EE.w, dtype=float).tolist(),
        'w_ie': np.array(W_IE.w, dtype=float).tolist()
    }
    if np.array(results[0]['w_ee'][-1]).mean(axis=0) < 0.1:
        return False
    elif np.array(results[0]['w_ie'][-1]).mean(axis=0) < 0.05:
        return False
    # We reset the spike monitors 
    del MPe_All, MPi_All, W_IE, W_EE
    # we run without monitoring
    b2.run(30 * second, report='text')
    # We set new spike and weight monitors 
    MPe_All = b2.SpikeMonitor(Pe)
    MPi_All = b2.SpikeMonitor(Pi)
    W_IE = b2.StateMonitor(con_ie, 'w', record=True, dt=0.5 * second)
    W_EE = b2.StateMonitor(con_ee, 'w', record=True, dt=0.5 * second)
    b2.run(120 * second, report='text')
    results[1] = {
        'start': 10,
        'end': 40,
        'ee_times': np.array(MPe_All.t / second, dtype=float).tolist(),  # Convert times to seconds
        'ee_neuron_ids': np.array(MPe_All.i, dtype=int).tolist(),
        'ie_times': np.array(MPi_All.t / second, dtype=float).tolist(),  # Convert times to seconds
        'ie_neuron_ids': np.array(MPi_All.i, dtype=int).tolist(),
        'w_ee': np.array(W_EE.w, dtype=float).tolist(),
        'w_ie': np.array(W_IE.w, dtype=float).tolist()
    }
    if np.array(results[1]['w_ee'][-1]).mean(axis=0) < 0.1:
        return False
    elif np.array(results[1]['w_ie'][-1]).mean(axis=0) < 0.05:
        return False
    # We get the information of the spike monitors

    # we save the results as well as return them 
    save_file = Path(working_directory) / "analysis_run_raw.json"
    logger.info(f"Saving to {save_file}")
    with open(save_file, "w") as f:
        json.dump(results, f)
    return True
