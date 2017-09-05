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

        self.add_function(
            "get_wave_form",
            call_cmd=self._get_wave_form,
            args=[Ints(min_value=0, max_value=1400)]
        )

    def _get_wave_form(self, n_samples):

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

    wave = dev.get_wave_form(1400)

    fig, ax = plt.subplots()
    ax.plot(*wave)
    plt.show()

if __name__ == "__main__":
    main()
