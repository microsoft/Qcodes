import numpy as np
import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers

class Demodulator:



    def __init__(self,
                 buffers_per_acquisition: int,
                 records_per_buffer: int,
                 samples_per_record: int,
                 sample_rate: float,
                 filter_settings,
                 active_channels):

        self.filter_settings = filter_settings
        self.active_channels = active_channels
        self.sample_rate = sample_rate
        settings = active_channels[0]
        if settings['average_buffers']:
            len_buffers = 1
        else:
            len_buffers = buffers_per_acquisition

        if settings['average_records']:
            len_records = 1
        else:
            len_records = records_per_buffer

        num_demods = 1
        demod_freqs = np.array(settings['demod_freq'])
        mat_shape = (num_demods, len_buffers,
                     len_records, samples_per_record)
        self.mat_shape = mat_shape
        integer_list = np.arange(samples_per_record)
        integer_mat = (np.outer(np.ones(len_buffers),
                                np.outer(np.ones(len_records), integer_list)))
        angle_mat = 2 * np.pi * \
                    np.outer(demod_freqs, integer_mat).reshape(mat_shape) / sample_rate
        self.cos_mat = np.cos(angle_mat)
        self.sin_mat = np.sin(angle_mat)


    def demodulate(self, volt_rec, int_delay, int_time):
        """
        Applies low bandpass filter and demodulation fit,
        and integration limits to samples array

        Args:
            record (numpy array): record from alazar to be multiplied
                                  with the software signal, filtered and limited
                                  to ifantegration limits shape = (samples_taken, )

        Returns:
            magnitude (numpy array): shape = (demod_length, samples_after_limiting)
            phase (numpy array): shape = (demod_length, samples_after_limiting)
        """

        # volt_rec to matrix and multiply with demodulation signal matrices
        demod_length = 1 #self.demod_freqs.get_num_demods()
        volt_rec_mat = np.outer(np.ones(demod_length), volt_rec).reshape(self.mat_shape)
        re_mat = np.multiply(volt_rec_mat, self.cos_mat)
        im_mat = np.multiply(volt_rec_mat, self.sin_mat)*0

        # filter out higher freq component
        cutoff = self.active_channels[0]['demod_freq']/10
        # self.demod_freqs.get_max_demod_freq() / 10
        if self.filter_settings['filter'] == 0:
            re_filtered = helpers.filter_win(re_mat, cutoff,
                                             self.sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=-1)
            im_filtered = helpers.filter_win(im_mat, cutoff,
                                             self.sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=-1)
        elif self.filter_settings['filter'] == 1:
            re_filtered = helpers.filter_ls(re_mat, cutoff,
                                            self.sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=-1)
            im_filtered = helpers.filter_ls(im_mat, cutoff,
                                            self.sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=-1)
        elif self.filter_settings['filter'] == 2:
            re_filtered = re_mat
            im_filtered = im_mat
        else:
            raise RuntimeError("Filter setting: {} not implemented".format(self.filter_settings['filter']))

        if self.active_channels[0]['integrate_samples']:
            # apply integration limits
            beginning = int(int_delay * self.sample_rate)
            end = beginning + int(int_time * self.sample_rate)

            re_limited = re_filtered[..., beginning:end]
            im_limited = im_filtered[..., beginning:end]
        else:
            re_limited = re_filtered
            im_limited = im_filtered

        # convert to magnitude and phase
        complex_mat = re_limited + im_limited * 1j
        magnitude = abs(complex_mat)
        phase = np.angle(complex_mat, deg=True)

        return magnitude, phase