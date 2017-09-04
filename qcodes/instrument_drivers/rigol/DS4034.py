import matplotlib.pyplot as plt


from qcodes import VisaInstrument, ArrayParameter


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

    def get_wave_form(self):
        self.data_format("asc")  # Make sure we get outputs in ascii format
        return self.visa_handle.query_ascii_values(":wav:data?", converter="s")  # We want to get the data as ascii
        # strings. Leaving query_ascii_values with a default converter=f will make it prone to exceptions as the
        # internal string to float function is not smart enough to deal with empty strings.


def main():

    visa_address = "TCPIP::169.254.180.17::INSTR"
    dev = Rigol_DS4035("rigol_scope", visa_address)

    data = dev.get_wave_form()
    data = [float(d) for d in data if d != ""]

    fig, ax = plt.subplots()
    ax.plot(data)
    plt.show()
    pass

if __name__ == "__main__":
    main()
