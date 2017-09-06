"""
The QCodes driver of the oscilloscope Rigol DS4034. Please see

http://int.rigol.com/File/TechDoc/20160831/DS4000E_ProgrammingGuide_EN.pdf

for the programming manual for this device
"""

import numpy as np
import matplotlib.pyplot as plt


from qcodes import VisaInstrument
from qcodes.utils.validators import Ints


class Rigol_DS4035(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        model = self.get_idn()['model']
        print("Established communication with device {}".format(model))

        self.add_parameter(
            "data_format",
            get_cmd=":wav:form?",
            set_cmd=":wav:form {}",
            val_mapping={
                'asc': 'ascii',
                'bin': 'binary'}
        )

        self.add_parameter(
            "time_base",
            get_cmd=":tim:scal?",
            get_parser=float,
            set_cmd=":tim:scal {}",
            unit="s/sample"
        )

        self.add_parameter(
            "sample_point_count",
            get_cmd=":wav:poin?",
            get_parser=int,
            set_cmd=":wav:poin {}",
            vals=Ints(min_value=0, max_value=1400)
        )

        self.add_parameter(
            "source_channel",
            set_cmd="WAV:SOUR CHAN{}",
            vals=Ints(min_value=0, max_value=4)
        )

        self.add_parameter(
            "mode",
            set_cmd=":wav:mode {}",
            val_mapping={
                'normal': 'norm'}
        )

        for channel_number in range(1, 5):  # It is really unfortunate that a get parameter does not except
            # arguments.
            # TODO: Improve the design of the Parameter class
            self.add_parameter(
                "measure_amplitude_channel{}".format(channel_number),
                get_cmd=":meas:vamp? chan{}".format(channel_number)
            )

            self.add_parameter(
                "vertical_scale_channel{}".format(channel_number),
                get_cmd="chan{}:scale?".format(channel_number),
                set_cmd="chan{}:scale ".format(channel_number) + "{}",
                get_parser=float
            )

        self.add_function(
            "get_wave_form",
            call_cmd=self._get_wave_form,
            args=[Ints(min_value=0, max_value=1400), Ints(min_value=0, max_value=4)]
        )

    def _get_wave_form(self, n_samples, channel):

        self.source_channel(channel)
        self.mode("normal")
        self.data_format("asc")
        self.sample_point_count(n_samples)

        data = self.visa_handle.query_ascii_values(":wav:data?", converter="s")
        # We want to get the data as ascii strings. Leaving query_ascii_values with a default converter=f will make
        # it prone to exceptions as the internal string to float function is not smart enough to deal with empty
        # strings.
        data = [float(d) for d in data if d != ""]

        times = self.time_base() * np.arange(0, len(data))

        return np.vstack([times, data])


def main():

    visa_address = "TCPIP::169.254.180.17::INSTR"
    dev = Rigol_DS4035("rigol_scope", visa_address)

    amp = dev.measure_amplitude_channel2()
    print("The amplitude of the signal is {}".format(amp))

    wave = dev.get_wave_form(1400, 2)

    fig, ax = plt.subplots()
    ax.plot(*wave)
    plt.show()

if __name__ == "__main__":
    main()
