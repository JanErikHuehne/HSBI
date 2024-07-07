import brian2 as b2
from brian2.units import *
from matplotlib import pyplot as plt
NE = 400
input_num = 40
input_freq = 30 
sim_time = 1
gmax = 20.0
lr = 1e-2
epsilon = 0.1
gl = 10 * nS
er = -80 * mV
el = -60 * mV
tau_gaba = 10*ms
tau_ampa = 10*ms
vt = -50*mV
memc = 200 * pfarad
eqs_neurons='''
                dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er)))/memc : volt (unless refractory)
                dg_ampa/dt = -g_ampa/tau_ampa : siemens
                dg_gaba/dt = -g_gaba/tau_gaba : siemens
            '''
neurons = b2.NeuronGroup(NE, model=eqs_neurons, threshold="v > vt",
                         reset="v=el", refractory=5*ms, method="euler")
neurons.v = -60*mV
con_ee = b2.Synapses(neurons,neurons, model="""w : 1""", on_pre="g_ampa += w*nS")

inhib = b2.NeuronGroup(100, model=eqs_neurons, threshold="v > vt",
                         reset="v=el", refractory=5*ms, method="euler")

con_ii = b2.Synapses(inhib,inhib, model="""w : 1""", on_pre="g_gaba += w*nS")
con_ii.connect(p=epsilon, condition='i != j')
con_ii.w = "rand()*0.1"


con_ie = b2.Synapses(inhib,neurons, model="""w : 1""", on_pre="g_gaba += w*nS")
con_ie.connect(p=epsilon, condition='i != j')
con_ie.w = "rand()*0.1"


con_ei = b2.Synapses(neurons,inhib, model="""w : 1""", on_pre="g_ampa += w*nS")
con_ei.connect(p=epsilon, condition='i != j')
con_ei.w = "rand()*0.5"


con_ee.connect(p=epsilon, condition='i != j')
con_ee.w = "rand()*0.5"

print(con_ee.w)
#P = b2.PoissonGroup(input_num, input_freq*Hz)
#S = b2.Synapses(P, neurons, on_pre='g_ampa += 10.0*nS').connect(p=0.3)
P = b2.PoissonInput(neurons, N=input_num, target_var="g_ampa", rate=25*Hz, weight=0.2*nS)
M = b2.SpikeMonitor(neurons)
M2 = b2.StateMonitor(neurons, 'v', record=True)



b2.run(sim_time * second)
t, i = M.it
b2.plot(t/ms, i, 'k.', ms=0.25)
b2.yticks([])
plt.savefig("/home/ge84yes/master_thesis/HSBI/images/test_network_1.png")
plt.clf()
b2.plot(M2.t/ms, M2.v[0]/mV, label='1')
b2.plot(M2.t/ms, M2.v[1]/mV, label='2')
plt.savefig("/home/ge84yes/master_thesis/HSBI/images/test_network.png")