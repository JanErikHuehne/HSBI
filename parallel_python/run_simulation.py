import argparse
from brian2 import *
def simulation(sim_params):
    start_scope()
    """Shared network parameters"""
    NE = 200
    NI = 160
    input_num = 100
    input_freq = 10 # Hz
    sim_time = 4
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
    neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                             reset='v=el', refractory=5*ms, method='euler')
    Pe = neurons[:NE]
    Pi = neurons[NE:]

    # EE plasticity 
    ee_alpha_pre = sim_params[0]
    ee_alpha_post = sim_params[1]
    ee_tauplus_stdp = sim_params[2] * ms
    ee_tauminus_stdp = sim_params[3] * ms
    ee_Aminus = -1.0
    ee_Aplus = ee_tauminus_stdp / ee_tauplus_stdp
    synapse_model ='''
                w : 1
                dee_trace_pre_plus/dt = -ee_trace_pre_plus / ee_tauplus_stdp : 1 (event-driven)
                dee_trace_pre_minus/dt = -ee_trace_pre_minus / ee_tauminus_stdp : 1 (event-driven)
                dee_trace_post_plus/dt = -ee_trace_post_plus / ee_tauplus_stdp : 1 (event-driven)
                dee_trace_post_minus/dt = -ee_trace_post_minus / ee_tauminus_stdp : 1 (event-driven)
    '''
    con_ee = Synapses(Pe, Pe, model=synapse_model,
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

    # EI Plasticity
    con_ei = Synapses(Pe, Pi, on_pre="g_ampa += 0.3*nS")
    con_ei.connect(p=epsilon)
    #  II Plasticity
    con_ii = Synapses(Pi,Pi, on_pre="g_gaba += 3*nS")
    con_ii.connect(p=epsilon)
    # IE Plasiticty 
    ie_alpha_pre = sim_params[4]
    ie_alpha_post = sim_params[5]
    ie_tauplus_stdp = sim_params[6] * ms
    ie_tauminus_stdp = sim_params[7] * ms
    ie_Aplus =  ie_tauminus_stdp / ie_tauplus_stdp
    ie_Aminus = -1.0
    synapse_model ='''
                w : 1
                die_trace_pre_plus/dt = -ie_trace_pre_plus / ie_tauplus_stdp : 1 (event-driven)
                die_trace_pre_minus/dt = -ie_trace_pre_minus / ie_tauminus_stdp : 1 (event-driven)
                die_trace_post_plus/dt = -ie_trace_post_plus / ie_tauplus_stdp : 1 (event-driven)
                die_trace_post_minus/dt = -ie_trace_post_minus / ie_tauminus_stdp : 1 (event-driven)
                '''
    con_ie.connect(p=epsilon)
    neurons.v = 0
    P = PoissonGroup(input_num, input_freq*Hz)
    S = Synapses(P, neurons, on_pre='g_ampa += 0.3*nS').connect(p=0.3)
    # Define monitors
    MPe = SpikeMonitor(Pe)
    MPi = SpikeMonitor(Pi)
    W_IE = StateMonitor(con_ie, 'w', record=True)
    W_EE = StateMonitor(con_ee, 'w', record=True)
    run(sim_time * second)
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
    logger.info("Successfully completed run!!")
    return  {'spikes': spikes, 'weights' : weights,
                 'runtime' : sim_time,
                 'plasticity' :list(str(k) for k in parameters.keys()),
                 't_start' : 0,#pre_simtime,
                 't_end' : sim_time,#pre_simtime+simtime,
                 'dt' : float(defaultclock.dt)}
        
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Add arguments
    parser.add_argument('parameters', metavar='simulation_parameters', type=float, nargs='+',
                        help='A list of parameters to process')

    # Parse the arguments
    args = parser.parse_args()
    sim_parameters = args.parameters
   
