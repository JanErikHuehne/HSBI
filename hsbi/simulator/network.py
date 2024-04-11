from brian2 import *
import numpy as np 
import multiprocessing


def parameter_order():
       return ['alpha_pre', 'alpha_post', 'Aplus', 'Aminus', 'tauplus_stdp', 'tauminus_stdp']

def plasticity_parameters(parameter_categories, parameter_list):
        """
        This function creates the plasticity parameter dictionary datatype as expected from the Rate_STDP_Network class.
        The function takes the input parameters:
        parameter_categories : list
                A list of the categories of plasticity. The categories must be one of the following: 'ee', 'ei', 'ie', 'ii'
        parameter_list : list
                This list contains all parameters. The total length of this list is 6*len(parameter_categories). 
                Parameters of different categories are expected to be listed after one another in the same order as they 
                appear in the parameter_categories list. 
                Each parameter set for each category itself is expected to have the following order:
                        alpha_pre : 0
                        alpha_post : 1
                        Aplus : 2
                        Aminus: 3
                        tauplus_stdp : 4
                        tauminus_stdp : 5
                The specific order of this parameters can be retrieved by the get_parameter_order function.
        """
        per_cat_param = 6
        param = {}
        for i,category in enumerate(parameter_categories):
                assert category in ['ee', 'ei', 'ie', 'ii'], "Specified category does not match a known plasticity category of the network. "
                cat_parameters = parameter_list[i*per_cat_param:(i+1)*per_cat_param]
                cat_dict = {'alpha_pre' : cat_parameters[0],
                 'alpha_post' : cat_parameters[1],
                 'Aplus' : cat_parameters[2],
                 'Aminus' : cat_parameters[3],
                 'tauplus_stdp' : cat_parameters[4],
                 'tauminus_stdp' : cat_parameters[5]
                }
                param[category] = cat_dict
        return param

class Rate_STDP_Network:
    def __init__(self, **kwargs):
       pass
    
    def run_simulation(self, parameters, **kwargs):
        start_scope()
        """Shared network parameters"""
        NE = 20
        NI = 5
        input_num = 100
        input_freq = 10 # Hz
        sim_time = 1 
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
        if parameters.get('ee'):
            ee_alpha_pre = parameters['ee']['alpha_pre']
            ee_alpha_post = parameters['ee']['alpha_post']
            ee_Aplus = parameters['ee']['Aplus']
            ee_Aminus = parameters['ee']['Aminus']
            ee_tauplus_stdp = parameters['ee']['tauplus_stdp'] * ms
            ee_tauminus_stdp = parameters['ee']['tauminus_stdp'] * ms
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
        else:
            con_ee = Synapses(Pe,Pe, on_pre="g_ampa += 3*nS")
        con_ee.connect(p=epsilon)

        # ###########################################
        #
        # This synpase model is the following
        #                   _ _
        #                  /   \
        #                 /  |  \
        #                /   |   \
        #               /    |    \
        #              /     |     \
        #     -  -  - / -  - |- -  -\  - - ->
        #     \ _ _  /       |       \ _ _ /
        #                    |
        #
        # 
        # ###########################################
        if parameters.get('ei'):
            # ###########################################
            # E-I Parametericed Plasticity 
            # ###########################################
            ei_alpha_pre = parameters['ei']['alpha_pre']
            ei_alpha_post = parameters['ei']['alpha_post']
            ei_Aplus = parameters['ei']['Aplus']
            ei_Aminus = parameters['ei']['Aminus']
            ei_tauplus_stdp = parameters['ei']['tauplus_stdp'] * ms
            ei_tauminus_stdp = parameters['ei']['tauminus_stdp'] * ms
            synapse_model ='''
                w : 1
                dei_trace_pre_plus/dt = -ei_trace_pre_plus / ei_tauplus_stdp : 1 (event-driven)
                dei_trace_pre_minus/dt = -ei_trace_pre_minus / ei_tauminus_stdp : 1 (event-driven)
                dei_trace_post_plus/dt = -ei_trace_post_plus / ei_tauplus_stdp : 1 (event-driven)
                dei_trace_post_minus/dt = -ei_trace_post_minus / ei_tauminus_stdp : 1 (event-driven)
            '''
            con_ei = Synapses(Pe, Pi, model=synapse_model,
                            on_pre='''
                                    g_ampa += w*nS
                                    ei_trace_pre_plus += 1.0
                                    ei_trace_pre_minus += 1.0
                                    w = clip(w + lr * (ei_alpha_pre + ei_Aplus * ei_trace_post_plus + ei_Aminus * ei_trace_post_minus), 0, gmax)
                                    ''',
                            on_post='''
                                    ei_trace_post_plus += 1
                                    ei_trace_post_minus += 1
                                    w = clip(w + lr * (ei_alpha_post + ei_Aplus * ei_trace_pre_plus + ei_Aminus * ei_trace_pre_minus), 0, gmax)
                                    '''
                                )
            
        else:
            con_ei = Synapses(Pe, Pi, on_pre="g_gaba += 3*nS")
        con_ei.connect(p=epsilon)
        if parameters.get('ii'):
            # ###########################################
            # I-I Parametericed Plasticity 
            # ###########################################
            ii_alpha_pre = parameters['ii']['alpha_pre']
            ii_alpha_post = parameters['ii']['alpha_post']
            ii_Aplus = parameters['ii']['Aplus']
            ii_Aminus = parameters['ii']['Aminus']
            ii_tauplus_stdp = parameters['ii']['tauplus_stdp'] * ms 
            ii_tauminus_stdp = parameters['ii']['tauminus_stdp'] * ms
            synapse_model ='''
                w : 1
                dii_trace_pre_plus/dt = -ii_trace_pre_plus / ii_tauplus_stdp : 1 (event-driven)
                dii_trace_pre_minus/dt = -ii_trace_pre_minus / ii_tauminus_stdp : 1 (event-driven)
                dii_trace_post_plus/dt = -ii_trace_post_plus / ii_tauplus_stdp : 1 (event-driven)
                dii_trace_post_minus/dt = -ii_trace_post_minus / ii_tauminus_stdp : 1 (event-driven)
            '''
            con_ii = Synapses(Pi, Pi, model=synapse_model,
                            on_pre='''
                                        g_gaba += w*nS
                                        ii_trace_pre_plus += 1.0
                                        ii_trace_pre_minus += 1.0
                                        w = clip(w + lr * (ii_alpha_pre + ii_Aplus * ii_trace_post_plus + ii_Aminus * ii_trace_post_minus), 0, gmax)
                                    ''',
                            on_post='''
                                        ii_trace_post_plus += 1
                                        ii_trace_post_minus += 1
                                        w = clip(w + lr * (ii_alpha_post + ii_Aplus * ii_trace_pre_plus + ii_Aminus * ii_trace_pre_minus), 0, gmax)
                                    '''
                            )
        else:
            con_ii = Synapses(Pi,Pi, on_pre="g_ampa += 3*nS")

        con_ii.connect(p=epsilon)
        if parameters.get('ie'):
            ie_alpha_pre = parameters['ie']['alpha_pre']
            ie_alpha_post = parameters['ie']['alpha_post']
            ie_Aplus = parameters['ie']['Aplus']
            ie_Aminus = parameters['ie']['Aminus']
            ie_tauplus_stdp = parameters['ie']['tauplus_stdp'] * ms
            ie_tauminus_stdp = parameters['ie']['tauminus_stdp'] * ms
            synapse_model ='''
                w : 1
                die_trace_pre_plus/dt = -ie_trace_pre_plus / ie_tauplus_stdp : 1 (event-driven)
                die_trace_pre_minus/dt = -ie_trace_pre_minus / ie_tauminus_stdp : 1 (event-driven)
                die_trace_post_plus/dt = -ie_trace_post_plus / ie_tauplus_stdp : 1 (event-driven)
                die_trace_post_minus/dt = -ie_trace_post_minus / ie_tauminus_stdp : 1 (event-driven)
                '''
            con_ie = Synapses(Pi, Pe, model=synapse_model,
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
        else:
            con_ie = Synapses(Pi,Pe, on_pre="g_ampa += 3*nS")
        con_ie.connect(p=epsilon)
        # Create neuron group
        neurons.v = 0

        P = PoissonGroup(input_num, input_freq*Hz)
        S = Synapses(P, neurons, on_pre='g_ampa += 0.3*nS').connect(p=0.3)
        # Define monitors
        MPe = SpikeMonitor(Pe)
        MPi = SpikeMonitor(Pi)

        weight_monitors = {}
        W_EI, W_IE, W_II, W_EE = None, None, None, None
        if parameters.get('ie'):
              W_IE = StateMonitor(con_ie, 'w', record=True)
        if parameters.get('ei'):
              W_EI = StateMonitor(con_ei, 'w', record=True)
        if parameters.get('ii'):
                W_II = StateMonitor(con_ii, 'w', record=True)
        if parameters.get('ee'):
               W_EE = StateMonitor(con_ee, 'w', record=True)

        spikes = {}
        # Run simulation
        run(sim_time * second)

        """
        Result Retrieval
        """
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
        if parameters.get('ie'):
               weights['ie'] = {
                                'from' :'Pi',
                                'to' : 'Pe',
                                'weights' : np.array(W_IE.w).copy()
                                }
        if parameters.get('ei'):
                 weights['ei'] = {
                                    'from' :'Pe',
                                    'to' : 'Pi',
                                    'weights' :  np.array(W_EI.w).copy()
                                    }
        if parameters.get('ii'):
                weights['ii'] = {
                                'from' :'Pi',
                                'to' : 'Pi',
                                'weights' :  np.array(W_II.w).copy()
                                }
        if parameters.get('ee'):
                weights['ee'] = {
                                'from' :'Pe',
                                'to' : 'Pe',
                                'weights' : np.array(W_EE.w).copy()
                                }
        return  {'spikes': spikes, 'weights' : weights,
                 'runtime' : sim_time,
                 'plasticity' :list(str(k) for k in parameters.keys()),
                 't_start' : 0,#pre_simtime,
                 't_end' : sim_time,#pre_simtime+simtime,
                 'dt' : float(defaultclock.dt)}

def run_simulation(args):
    model = Rate_STDP_Network(**args)
    return model.run_simulation(**args)





def run_sim():
        plasticity_categories = ['ie', 'ee']
        # return ['alpha_pre', 'alpha_post', 'Aplus', 'Aminus', 'tauplus_stdp', 'tauminus_stdp']
        plasticity_params = [-0.1, -1., 3.3, 2.0, 4.0, 3.9, -0.1, -1., 3.3, 2.0, 4.0, 3.9]
        plasticity_params = plasticity_parameters(plasticity_categories, plasticity_params)
        network_params = Network_Parameters()
        sim_res = Rate_STDP_Network.run(network_params, plasticity_params)

        metrics = Metrics(broad=True)
        print(metrics.calculate_metrics(sim_res))