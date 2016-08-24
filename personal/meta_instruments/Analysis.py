import numpy as np
import peakutils
from matplotlib import pyplot as plt

from qcodes import Instrument
import qcodes.instrument.parameter as parameter
from qcodes.utils import validators as vals


class BasicAnalysis(Instrument):
    shared_kwargs = ['ATS_controller']

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def find_high_low(self, traces, plot=False, threshold_peak=0.02):
        hist, bin_edges = np.histogram(np.ravel(traces), bins=30)

        # Find two peaks
        for k in range(4):
            peaks_idx = np.sort(peakutils.indexes(hist, thres=threshold_peak, min_dist=5))
            if len(peaks_idx) == 2:
                print('2 peaks found')
                break
            elif len(peaks_idx) == 1:
                print('One peak found instead of two, lowering threshold')
                threshold_peak /= 1.5
            elif len(peaks_idx) > 2:
                print('Found {} peaks instead of two, increasing threshold'.format(len(peaks_idx)))
                threshold_peak *= 1.5
            else:
                print('No peaks found')
                return None, None, None

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
        if SNR < 3:
            'Signal to noise ratio {} is too low'.format(SNR)
            threshold_voltage = None

        # Plotting
        if plot:
            plt.figure()
            for k, signal in enumerate([low, high]):
                sub_hist, sub_bin_edges = np.histogram(np.ravel(signal['traces']), bins=10)
                plt.bar(sub_bin_edges[:-1], sub_hist, width=0.05, color='bg'[k])
                plt.plot(signal['mean'], hist[peaks_idx[k]], 'or', ms=12)

        return low, high, threshold_voltage

    def edge_voltage(self, traces, edge, state, threshold_voltage=None, points=4, plot=False):
        assert edge in ['begin', 'end'], 'Edge {} must be either "begin" or "end"'.format(edge)
        assert state in ['low', 'high'], 'State {} must be either "low" or "high"'.format(state)
        idx_list = slice(None, 4) if edge == 'begin' else slice(-4, None)

        # Determine threshold voltage if not provided
        if threshold_voltage is None:
            low, high, threshold_voltage = self.find_high_low(traces, plot=plot)

        if threshold_voltage is None:
            print('Could not find two peaks for empty and load state')
            success = np.array([False] * len(traces))
        elif state == 'low':
            success = [np.mean(trace[idx_list]) < threshold_voltage for trace in traces]
        else:
            success = [np.mean(trace[idx_list]) > threshold_voltage for trace in traces]
        return np.array(success)

    def find_up_proportion(self, traces, threshold_voltage, return_mean=True, start_point=50, filter_window=0, plot=False):
        # trace has to contain read stage only
        # TODO Change start point to start time (sampling rate independent)
        if not threshold_voltage:
            _, _, threshold_voltage = self.find_high_low(traces, plot=plot)

        if filter_window > 0:
            traces=[np.convolve(trace, np.ones(filter_window) / filter_window, mode='valid') for trace in traces]
            # from scipy import signal
            # savgol_filter = signal.savgol_filter
            # traces=[savgol_filter(trace,window_length=19,polyorder=4) for trace in traces]

        # Filter out the traces that contain one or more peaks
        traces_up_electron = [np.any(trace[start_point:] > threshold_voltage) for trace in traces]

        if return_mean:

            return sum(traces_up_electron) / len(traces)
        else:
            return traces_up_electron

    def measure_up_proportion(self):
        self.ATS_controller.average_mode('none')
        self._traces = self.ATS_controller.acquisition()
        return self.find_up_proportion(traces=self._traces)

class EmptyLoadReadAnalysis(BasicAnalysis):
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

        self.add_parameter(name='traces',
                           get_cmd=lambda: self._traces,
                           vals=vals.Anything())

        self._traces = None

    def _fidelity(self):
        self.ATS_controller.average_mode('none')
        self._traces = self.ATS_controller.acquisition()
        return self.analyse_traces(self._traces[0])

    def analyse_traces(self, traces, plot=False):
        ATS_sample_rate = self.ATS_controller._get_alazar_parameter('sample_rate')

        load_pts = round(self.load_duration() / 1e3 * ATS_sample_rate)
        empty_pts = round(self.empty_duration() / 1e3 * ATS_sample_rate)
        read_pts = round(self.read_duration() / 1e3 * ATS_sample_rate)

        traces_empty = traces[:, :empty_pts]
        traces_load = traces[:, empty_pts:empty_pts + load_pts]
        traces_read = traces[:, empty_pts + load_pts:empty_pts + load_pts + read_pts]

        fidelity_empty = self.analyse_empty(traces_empty, plot=plot)
        fidelity_load = self.analyse_load(traces_load, plot=plot)
        fidelity_read = 1 - self.analyse_read(traces_read, plot=plot) / read_pts

        return fidelity_load, fidelity_read, fidelity_empty

    def analyse_read(self, traces, plot=False, return_mean=True):
        low, high, threshold_voltage = self.find_high_low(traces, plot=plot)

        if threshold_voltage is None:
            print('Could not find two peaks for empty and load state')
            # Return the full trace length as mean if return_mean=True
            return traces.shape[1] if return_mean else []

        # Filter out the traces that start off loaded
        idx_begin_loaded = self.edge_voltage(traces, edge='begin', state='low',
                                             threshold_voltage=threshold_voltage)
        traces_loaded = traces[idx_begin_loaded]

        if not len(idx_begin_loaded):
            print('None of the load traces start with an loaded state')
            return traces.shape[1] if return_mean else []

        # Filter out the traces that at some point have conductance
        # Assume that if there is current, the electron must have been up
        final_conductance_idx_list = [max(np.where(trace > threshold_voltage)[0])
                                      for trace in traces_loaded
                                      if np.max(trace) > threshold_voltage]
        if return_mean:
            return np.mean(final_conductance_idx_list)
        else:
            return final_conductance_idx_list

    def analyse_load(self, traces, plot=False, return_idx=False):
        idx_list = np.arange(len(traces))
        low, high, threshold_voltage = self.find_high_low(traces, plot=plot)

        if threshold_voltage is None:
            print('Could not find two peaks for empty and load state')
            return 0 if not return_idx else 0, []

        # Filter data that starts at high conductance (no electron)
        idx_begin_empty = self.edge_voltage(traces, edge='begin', state='high',
                                            threshold_voltage=threshold_voltage)
        traces_begin_empty = traces[idx_begin_empty]
        idx_list = idx_list[idx_begin_empty]

        if not len(idx_begin_empty):
            print('None of the load traces start with an empty state')
            return 0 if not return_idx else 0, []

        idx_end_load = self.edge_voltage(traces_begin_empty, edge='end', state='low',
                                         threshold_voltage=threshold_voltage)

        idx_list = idx_list[idx_end_load] if len(idx_end_load) else []

        if return_idx:
            return sum(idx_end_load) / sum(idx_begin_empty), idx_list
        else:
            return sum(idx_end_load) / sum(idx_begin_empty)

    def analyse_empty(self, traces, plot=False, return_idx=False):
        idx_list = np.arange(len(traces))
        low, high, threshold_voltage = self.find_high_low(traces, plot=plot)

        if threshold_voltage is None:
            print('Could not find two peaks for empty and load state')
            return 0

        # Filter data that starts at high conductance (no electron)
        idx_begin_load = self.edge_voltage(traces, edge='begin', state='low',
                                           threshold_voltage=threshold_voltage)
        traces_begin_load = traces[idx_begin_load]
        idx_list = idx_list[idx_begin_load]

        if not len(idx_begin_load):
            print('None of the empty traces start with a loaded state')
            return 0

        idx_end_empty = self.edge_voltage(traces_begin_load, edge='end', state='high',
                                          threshold_voltage=threshold_voltage)

        idx_list = idx_list[idx_end_empty] if len(idx_end_empty) else []

        if return_idx:
            return sum(idx_end_empty) / sum(idx_begin_load), idx_list
        else:
            return sum(idx_end_empty) / sum(idx_begin_load)
