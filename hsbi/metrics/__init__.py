# This sub-libary should contain all the necessary code to compute different metrics 
# given the raw output of spike trains from the simulation.
from ..utils import parameter
import numpy as np
from scipy.signal import correlate

class Metrics:
    """
    Abstract Class for all Metrics
    """
    def __init__(self, broad:bool, wmax=20.0):
        """
        Meta Information of Vogles paper 

        Attribute: NE
        Value: 4096
        Attribute: NI
        Value: 1024
        Attribute: N_input
        Value: 5000
        Attribute: ampa_nmda_ratio
        Value: 0.3
        Attribute: bin_size_big
        Value: 0.1
        Attribute: bin_size_fft
        Value: 0.01
        Attribute: bin_size_medium
        Value: 0.01
        Attribute: bin_size_small
        Value: 0.001
        Attribute: blow_up
        Value: -1.0
        Attribute: eta
        Value: 0.01
        Attribute: isi_lim_kl_isi
        Value: [0 1]
        Attribute: lns
        Value: 60
        Attribute: ls
        Value: 10
        Attribute: max_rate_checker
        Value: 100
        Attribute: n_bins_kl_isi
        Value: 100
        Attribute: n_recorded
        Value: 1000
        Attribute: n_recorded_i
        Value: 500
        Attribute: name
        Value: sim_bg_IF_EEEIIEII_6pPol
        Attribute: record_i
        Value: True
        Attribute: sparseness
        Value: 0.1
        Attribute: sparseness_poisson
        Value: 0.05
        Attribute: t_start_rec
        Value: 60
        Attribute: t_stop_rec
        Value: 70
        Attribute: tau_ampa
        Value: 0.005
        Attribute: tau_checker
        Value: 1
        Attribute: tau_gaba
        Value: 0.01
        Attribute: tau_nmda
        Value: 0.1
        Attribute: wee
        Value: 0.1
        Attribute: wei
        Value: 0.1
        Attribute: weight_poisson
        Value: 0.075
        Attribute: wie
        Value: 1
        Attribute: wii
        Value: 1
        Attribute: window_view_auto_cov
        Value: 0.5
        Attribute: wmax
        Value: 20
        """
        self.broad = broad
        self.wmax = wmax
        self.bin_size_big = 0.1
        self.bin_size_medium = 0.01
        self.bin_size_small = 0.001
        self.window_view_auto_cov = 0.5

    def extract_neuron_spikes(self, spike_times, neuron_ids):
            neuron_spikes = {}
            for i in range(len(neuron_ids)):
                neuron_id = neuron_ids[i]
                if not neuron_spikes.get(neuron_id):
                    neuron_spikes[neuron_id] = []
                neuron_spikes[neuron_id].append(spike_times[i])
            return neuron_spikes
    def rate_i(self, sim_data):
        ####################################
        # This method computes the global firing rate of the inhibitory neuron population
        # We access the spike data of the inibitory neurons
        ####################################

        spike_data = sim_data['spikes']['Pi']
        num_neurons = spike_data['num_neurons']

        # spike times
        spike_times = spike_data['times']

        total_num_of_spikes = len(spike_times)

        # firing rate
        rate = total_num_of_spikes / (num_neurons * sim_data['runtime'])
        return rate


    def rate_e(self, sim_data):
        ####################################
        # This method computes the global firing rate of the inhibitory neuron population
        # We access the spike data of the inibitory neurons
        ####################################

        spike_data = sim_data['spikes']['Pe']
        num_neurons = spike_data['num_neurons']

        # spike times
        spike_times = spike_data['times']

        total_num_of_spikes = len(spike_times)

        # firing rate
        rate = total_num_of_spikes / (num_neurons * sim_data['runtime'])
        return rate

    def weif(self, sim_data):
        """final mean EI weight"""
        w_trace = sim_data['weights']['ei']['weights']
        return np.mean(w_trace[:, -1])
    
    def weef(self, sim_data):
        """final mean EE weight"""
        w_trace = sim_data['weights']['ee']['weights']
        return np.mean(w_trace[:, -1])
         

    def wief(self, sim_data):
        """final mean IE weight"""
        w_trace = sim_data['weights']['ie']['weights']
        return np.mean(w_trace[:, -1])
        

    def wiif(self, sim_data):
        """final mean II weight"""
        w_trace = sim_data['weights']['ii']['weights']
        return np.mean(w_trace[:, -1])
    
    def w_creep(self, sim_data):
        """creep of the weights
            Following to Confavreux et al. (2023)
            Change of the mean weight between start and finish  as percentage,
            taken as the maximum compared between different plasticity connections
        """
        plasticity = sim_data['plasticity']
        max_creep = 0.0
        for conn in plasticity:
            w_trace = sim_data['weights'][conn]['weights']
            start_w = np.mean(w_trace[:, 0])
            end_w = np.mean(w_trace[:, -1])
            if start_w + end_w > 0.1:
                creep_c = np.abs(2* (end_w - start_w) / (end_w + start_w))
                if creep_c > max_creep:
                    max_creep = creep_c
        return max_creep
    
    def w_blow(self, sim_data):
        """Percentage of weights that did blow up to a maximum value"""
        plasticity = sim_data['plasticity']
        f_blow = 0.0
        for conn in plasticity:
            w_trace = sim_data['weights'][conn]['weights']
            blow = np.sum(w_trace > self.wmax)
            vanish = np.sum(w_trace < 0.001)
            f_blow += blow / w_trace.size
            f_blow += vanish / w_trace.size
        return f_blow / len(plasticity)
    
    def cv_isi(self, sim_data):
        """coefficient of variation of the interspike intervals of exitatory neuron poulation"""

        
        spike_data = sim_data['spikes']['Pe']
        spike_times = spike_data['times']
        neuron_ids = spike_data['neurons']
        grouped_spikes = self.extract_neuron_spikes(spike_times,neuron_ids)
        
        var_isi_val = []
        for key,val in zip(grouped_spikes.keys(), grouped_spikes.values()):
            isi = np.std(np.diff(val)) / np.mean(np.diff(val))
            var_isi_val.append(isi)
        return np.mean(var_isi_val)

    def averaged_fano_spatial(self, sim_data):
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'],self.bin_size_big)
    
        grouped_spikes_e = self.extract_neuron_spikes(sim_data['spikes']['Pe']['times'], sim_data['spikes']['Pe']['neurons'])
        grouped_spikes_i = self.extract_neuron_spikes(sim_data['spikes']['Pi']['times'], sim_data['spikes']['Pi']['neurons'])
        binned = np.empty(shape=(0, len(tbins)-1))
        
        # exitatory
        for spike_train in grouped_spikes_e.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned_ebinnedxitatory = np.vstack([binned, binned_spike_train])

        # inhibitory
        for spike_train in grouped_spikes_i.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned = np.vstack([binned, binned_spike_train])
        
        # compute the fano factor over each time window 
        mean = np.mean(binned, axis=0)
        var = np.var(binned, axis=0)

        fano = var / mean
        return np.mean(fano)
    
    def averaged_fano_time(self, sim_data):
        """fano factor for each spike train averaged over the population
            Binning the spike-trains over 100ms successive windows 
            Fano Factor computed per neuron, then averaged over the population
        """
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'],self.bin_size_big)

        grouped_spikes_e = self.extract_neuron_spikes(sim_data['spikes']['Pe']['times'], sim_data['spikes']['Pe']['neurons'])
        grouped_spikes_i = self.extract_neuron_spikes(sim_data['spikes']['Pi']['times'], sim_data['spikes']['Pi']['neurons'])
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
    

    def std_fr_s(self, sim_data):
        """
        Standard deviation of the firing rate over spacial domain  
        """

     
        grouped_spikes_e = self.extract_neuron_spikes(sim_data['spikes']['Pe']['times'], sim_data['spikes']['Pe']['neurons'])
        grouped_spikes_i = self.extract_neuron_spikes(sim_data['spikes']['Pi']['times'], sim_data['spikes']['Pi']['neurons'])
       
        all_rates = []
        # exitatory
        for spike_train in grouped_spikes_e.values():
          all_rates.append(len(spike_train) / sim_data['runtime'])

        # inhibitory
        for spike_train in grouped_spikes_i.values():
            all_rates.append(len(spike_train) / sim_data['runtime'])
        
        return np.std(np.array(all_rates))
    
    def std_fr(self, sim_data):
        """standard deviation of the firing rate
            FR are computed over successive 1ms time windows, on which the standard deviation was computed
        """
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'],self.bin_size_medium)
        grouped_spikes_e = self.extract_neuron_spikes(sim_data['spikes']['Pe']['times'], sim_data['spikes']['Pe']['neurons'])
        grouped_spikes_i = self.extract_neuron_spikes(sim_data['spikes']['Pi']['times'], sim_data['spikes']['Pi']['neurons'])
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
    

    def fft(self, sim_data):
        """ fourier transform, aread under the curve
        spike trains are 
        
        """
    def auto_cov(self, sim_data):
        """autocovariance of the spike train
            Spike trains are binned in a window of 10 ms
            auto-covariance for each neuron is computed and normalized between -500 ms and 500 ms
            Mean area under the curve is computed and averaged over neurons
        """

        # First we bin the spike trains
        grouped_spikes_e = self.extract_neuron_spikes(sim_data['spikes']['Pe']['times'], sim_data['spikes']['Pe']['neurons'])
        grouped_spikes_i = self.extract_neuron_spikes(sim_data['spikes']['Pi']['times'], sim_data['spikes']['Pi']['neurons'])
        # We iterate through the grouped spike trains and as a first step bin them 
        tbins = np.arange(sim_data['t_start'], sim_data['t_end'], self.bin_size_medium)
        binned = np.empty(shape=(0, len(tbins)-1))
        

        # exitatory
        for spike_train in grouped_spikes_e.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned_ebinnedxitatory = np.vstack([binned, binned_spike_train])

        # inhibitory
        for spike_train in grouped_spikes_i.values():
            binned_spike_train = np.histogram(spike_train, tbins)[0]
            binned = np.vstack([binned, binned_spike_train])
        
        # Correlate
        lags = int(self.window_view_auto_cov / self.bin_size_medium)
        aoc = []
        for x in binned:
            x_corr = correlate(x - x.mean(), x- x.mean(), mode='full')
            # normalize
            x_corr = np.abs(x_corr) / x_corr.max()
            aoc.append(np.mean(x_corr[(x_corr.size//2-(lags)):(x_corr.size//2+(lags+1))]))
        return np.mean(aoc)
       
    def calculate_metrics(self, sim_data, return_type):
        """
        This method calculates the metrics for the given simulation data
        """
        metrics = {}
        metrics['cv_isi'] = self.cv_isi(sim_data = sim_data)
        metrics['rate_i'] = self.rate_i(sim_data = sim_data)
        metrics['rate_e'] = self.rate_e(sim_data = sim_data)
        metrics['wmean_ie'] = self.wief(sim_data = sim_data)
        metrics['wmean_ee'] = self.weef(sim_data = sim_data)
        metrics['wcreep'] = self.w_creep(sim_data = sim_data)
        metrics['cv_isi'] = self.cv_isi(sim_data = sim_data)
        metrics['f_w-blow'] = self.w_blow(sim_data = sim_data) 
        metrics['mean_fano_t'] = self.averaged_fano_time(sim_data = sim_data)
        metrics['mean_fano_s'] = self.averaged_fano_spatial(sim_data = sim_data)
        metrics['std_fr'] = self.std_fr(sim_data = sim_data)
        metrics['std_fr_s'] = self.std_fr_s(sim_data = sim_data)
        metrics['auto_cov'] = self.auto_cov(sim_data = sim_data)
        if return_type == 'dict':
            return metrics
        elif return_type == "list":
            return list(metrics.values())

    
