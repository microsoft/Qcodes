import numpy as np
import peakutils
from matplotlib import pyplot as plt

from qcodes import Instrument
import qcodes.instrument.parameter as parameter


class BasicAnalysis(Instrument):
    shared_kwargs = ['ATS_controller']

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def find_high_low(self, traces, plot=False):
        hist, bin_edges = np.histogram(np.ravel(traces), bins=30)
        peaks_idx = np.sort(peakutils.indexes(hist, thres=0.02, min_dist=5))
        assert len(peaks_idx) == 2, 'Found {} peaks instead of two'.format(len(peaks_idx))

        # Find threshold, mean low, and mean high voltages
        threshold_idx = int(round(np.mean(peaks_idx)))
        threshold_voltage = bin_edges[threshold_idx]

        # Create dictionaries containing information about the low, high state
        low, high = {}, {}
        low['traces'] = traces[traces < threshold_voltage]
        high['traces'] = traces[traces > threshold_voltage]
        for signal in [low, high]:
            signal['mean'] = np.mean(signal['traces'])
            signal['std'] = np.std(signal['traces'])
        SNR = (high['mean'] - low['mean']) / np.sqrt(high['std'] ** 2 + low['std'] ** 2)
        assert SNR > 3, 'Signal to noise ratio {} is too low'.format(SNR)

        # Plotting
        if plot:
            plt.figure()
            for k, signal in enumerate([low, high]):
                sub_hist, sub_bin_edges = np.histogram(np.ravel(signal['data']), bins=10)
                plt.bar(sub_bin_edges[:-1], sub_hist, width=0.05, color='bg'[k])
                plt.plot(signal['mean'], 100, 'or', ms=12)

            plt.plot(bin_edges[:-1][peaks_idx], hist[peaks_idx], 'or', ms=12)
        return low, high, threshold_voltage

    def edge_voltage(self, traces, edge, state, points=4):
        assert edge in ['begin', 'end'], 'Edge {} must be either "begin" or "end"'.format(edge)
        assert state in ['low', 'high'], 'State {} must be either "low" or "high"'.format(state)
        idx_list = slice(None, 4) if edge == 'begin' else slice(-4, None)

        low, high, threshold_voltage = self.find_high_low(traces)
        if state == 'low':
            success = [np.mean(trace[idx_list]) < threshold_voltage for trace in traces]
        else:
            success = [np.mean(trace[idx_list]) > threshold_voltage for trace in traces]
        return success


class LoadReadEmptyAnalysis(BasicAnalysis):
    shared_kwargs = ['ATS_controller']

    def __init__(self, name, ATS_controller, **kwargs):
        super().__init__(name, **kwargs)
        self.ATS_controller = ATS_controller

        self.add_parameter(name='load_duration',
                           units='ms',
                           parameter_class=parameter.ManualParameter)
        self.add_parameter(name='read_duration',
                           units='ms',
                           parameter_class=parameter.ManualParameter)
        self.add_parameter(name='empty_duration',
                           units='ms',
                           parameter_class=parameter.ManualParameter)

        self.add_parameter(name='fidelity',
                           names=['load_fidelity', 'read_fidelity', 'empty_fidelity'],
                           get_cmd=self._fidelity)

    def _fidelity(self):
        self.ATS_controller.average_mode('none')
        traces = self.ATS_controller.acquisition()
        return self.analyse_traces(traces[0])

    def analyse_traces(self, traces):
        ATS_sample_rate = self.ATS_controller._get_alazar_parameter('sample_rate')

        load_pts = round(self.load_duration() / 1e3 * ATS_sample_rate)
        read_pts = round(self.read_duration() / 1e3 * ATS_sample_rate)
        empty_pts = round(self.empty_duration() / 1e3 * ATS_sample_rate)

        traces_load = traces[:, :load_pts]
        traces_read = traces[:, load_pts:load_pts + read_pts]
        traces_empty = traces[:, load_pts + read_pts:]

        fidelity_load = self.analyse_load(traces_load)
        fidelity_read = self.analyse_read(traces_read) / read_pts
        fidelity_empty = self.analyse_empty(traces_empty)

        return fidelity_load, fidelity_read, fidelity_empty

    def analyse_read(self, traces):
        low, high, threshold_voltage = self.find_high_low(traces)

        # Filter out the traces that start off
        idx_loaded = self.edge_voltage(traces, edge='begin', state='low')
        traces_loaded = traces[np.array(idx_loaded)]

        # Filter out the traces that at some point have conductance
        # Assume that if there is current, the electron must have been up
        final_conductance_idx_list = [max(np.where(trace > threshold_voltage)[0])
                                      for trace in traces_loaded
                                      if np.max(trace) > threshold_voltage]
        return np.mean(final_conductance_idx_list)

    def analyse_load(self, traces):
        # Filter data that starts at high conductance (no electron)
        idx_begin_empty = self.edge_voltage(traces, edge='begin', state='high')
        traces_begin_empty = traces[np.array(idx_begin_empty)]

        idx_end_load = self.edge_voltage(traces_begin_empty, edge='end', state='low')
        return sum(idx_end_load) / sum(idx_begin_empty)

    def analyse_empty(self, traces):
        # Filter data that starts at high conductance (no electron)
        idx_begin_load = self.edge_voltage(traces, edge='begin', state='low')
        traces_begin_load = traces[np.array(idx_begin_load)]

        idx_end_empty = self.edge_voltage(traces_begin_load, edge='end', state='high')
        return sum(idx_end_empty) / sum(idx_begin_load)
