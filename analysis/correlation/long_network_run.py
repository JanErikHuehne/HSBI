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
from pathlib import Path
b2.prefs.codegen.target = "numpy"




def test_run(sim_params):
    NE = 400
    NI = NE // 4
    input_num = 40
    input_freq = 30 # Hz
    sim_time = 120 * ms
    gmax = 20.0
    lr = 1e-2 
    Aminus = 1.0
    epsilon = 0.2
    gl = 10 * nS
    er = -80 * mV
    el = -60 * mV
    tau_gaba = 10.0 * ms
    tau_ampa = 5.0 * ms
    vt = -50 * mV
    memc = 200 * pfarad
    # Neuron equations
    eqs_neurons = '''
        dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gaba/dt = -g_gaba/tau_gaba : siemens
        '''
    # Create neuron groups
    Pe = b2.NeuronGroup(NE, model=eqs_neurons, threshold='v > vt',
                            reset='v=el', refractory=5*ms, method='euler')
    Pi = b2.NeuronGroup(NI, model=eqs_neurons, threshold='v > vt',
                            reset='v=el', refractory=5*ms, method='euler')
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
                            w = clip(w + lr * (ee_alpha_pre + factor_ee*ee_Aplus*ee_trace_post_plus + factor_ee*ee_Aminus * ee_trace_post_minus), 0, gmax)
                            ''',
                        on_post='''
                            ee_trace_post_plus += 1
                            ee_trace_post_minus += 1
                            w = clip(w + lr * (ee_alpha_post + ee_Aplus*ee_trace_pre_plus + ee_Aminus * ee_trace_pre_minus), 0, gmax)
                            '''
                        )
    con_ee.connect(p=epsilon, condition='i != j')
    con_ee.w = 0.2
    con_ei = b2.Synapses(Pe, Pi, on_pre="g_ampa += 0.2*nS")
    con_ei.connect(p=epsilon, condition='i != j')
    # II Plasticity
    con_ii = b2.Synapses(Pi, Pi, on_pre="g_gaba += 1*nS")
    con_ii.connect(p=epsilon, condition='i != j')
    #con_ie = b2.Synapses(Pi, Pe, on_pre="g_ampa += 0.2*nS")
    #con_ie.connect(p=epsilon, condition='i != j')
    P = b2.PoissonGroup(input_num, input_freq * Hz)
    S = b2.PoissonInput(Pe, N=input_num, target_var="g_ampa", rate=20 * Hz, weight=0.4 * nS)
    b2.run(10 * second, report='text')
   

def network_run(sim_params, working_directory):
    logger.info(f"Running in {working_directory}")
    b2.start_scope()

    # Shared network parameters
    NE = 400
    NI = NE // 4
    input_num = 40
    input_freq = 30 # Hz
    sim_time = 120 * ms
    gmax = 20.0
    lr = 1e-2 
    Aminus = 1.0
    epsilon = 0.2
    gl = 10 * nS
    er = -80 * mV
    el = -60 * mV
    tau_gaba = 10.0 * ms
    tau_ampa = 5.0 * ms
    vt = -50 * mV
    memc = 200 * pfarad

    # Neuron equations
    eqs_neurons = '''
        dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gaba/dt = -g_gaba/tau_gaba : siemens
        '''

    # Create neuron groups
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
                            w = clip(w + lr * (ee_alpha_pre + factor_ee*ee_Aplus*ee_trace_post_plus + factor_ee*ee_Aminus * ee_trace_post_minus), 0, gmax)
                            ''',
                        on_post='''
                            ee_trace_post_plus += 1
                            ee_trace_post_minus += 1
                            w = clip(w + lr * (ee_alpha_post + ee_Aplus*ee_trace_pre_plus + ee_Aminus * ee_trace_pre_minus), 0, gmax)
                            '''
                        )
    con_ee.connect(p=epsilon, condition='i != j')
    con_ee.w = 0.2

    # EI Plasticity - Add minimal synapse model
    con_ei = b2.Synapses(Pe, Pi, model='''w : 1''', on_pre="g_ampa += 0.2*nS")
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
                            w = clip(w + lr * (ie_alpha_pre + factor_ie*ie_Aplus * ie_trace_post_plus + factor_ie*ie_Aminus * ie_trace_post_minus), 0, gmax)
                            ''',
                        on_post='''
                            ie_trace_post_plus += 1
                            ie_trace_post_minus += 1
                            w = clip(w + lr * (ie_alpha_post + ie_Aplus * ie_trace_pre_plus + ie_Aminus * ie_trace_pre_minus), 0, gmax)
                            '''
                        )
    con_ie.connect(p=epsilon, condition='i != j')
    con_ie.w = 1.0

    # Initialize neurons' membrane potential
    neurons.v = el

    # Input groups
    P = b2.PoissonGroup(input_num, input_freq * Hz)
    S = b2.PoissonInput(Pe, N=input_num, target_var="g_ampa", rate=20 * Hz, weight=0.4 * nS)
    # Pre-run of 60 seconds
    b2.run(5 * second, report='text')
    # First Recording Window
    MPe_All = b2.SpikeMonitor(Pe)
    MPi_All = b2.SpikeMonitor(Pi)
    W_IE = b2.StateMonitor(con_ie, 'w', record=True, dt=30 * second)
    W_EE = b2.StateMonitor(con_ee, 'w', record=True, dt=30 * second)
    b2.run(30 * second, report='text')
    # We get the information of the spike monitors
    results = {
        'start': 10,
        'end': 40,
        'ee_times': np.array(MPe_All.t / second, dtype=float).tolist(),  # Convert times to seconds
        'ee_neuron_ids': np.array(MPe_All.i, dtype=int).tolist(),
        'ie_times': np.array(MPi_All.t / second, dtype=float).tolist(),  # Convert times to seconds
        'ie_neuron_ids': np.array(MPi_All.i, dtype=int).tolist(),
        'w_ee': np.array(W_EE.w, dtype=float).tolist(),
        'w_ie': np.array(W_IE.w, dtype=float).tolist(),
        "ee_pre": np.array(con_ee.i).tolist(),
        "ee_post" : np.array(con_ee.j).tolist(),
        "ie_pre": np.array(con_ie.i).tolist(),
        "ie_post" : np.array(con_ie.j).tolist(),
    }


    # we save the results as well as return them 
    # create long run directory 
    working_directory = Path(working_directory)
    working_directory = working_directory / "long"
    working_directory.mkdir(exist_ok=True)
    save_file = Path(working_directory) / "long_run.json"
    logger.info(f"Saving to {save_file}")
    with open(save_file, "w") as f:
        json.dump(results, f)
    return True


if __name__ == '__main__':
    data_path = "/home/ge84yes/data/run_2/analysis/analysis/f269f398ab4e3d3b421b3a24e9654a066ae3b23a8efd7ed5b1dd564f49a17c97"
    # we load the parameters
    data_path = Path(data_path)
    parameter_file = data_path / 'parameters.json'
    with open(parameter_file, 'r') as f:
        parameter_dict = json.load(f)
    parameters = list(parameter_dict.values())
    parameters = [float(s) for s in parameters]
    print(parameters)
    test_run(sim_params=parameters)
    # Start the long run
  

    
