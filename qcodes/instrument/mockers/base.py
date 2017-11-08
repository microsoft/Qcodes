# Far future TODO: make the fake instrument a virtual VISA resource and
# don't change ANYTHING in the driver under test


class FakeSCPIInstrument:
    """
    Class for mocking an instrument (NOT the driver)
    """
    def __init__(self) -> None:

        # the initialise_state should be overwritten by subclasses
        # where this is needed, e.g. if the IDN string contains a
        # model number that is used by the driver
        self.initialise_state()

    def initialise_state(self):
        self.state = {'*IDN?': 'QCoDeS, b0gu5, 1337, 0.0.0'}

    def __call__(self, cmd: str) -> None:
        """
        A mocking of the SCPI protocol
        """

        # set/get logic
        # we will eventually have to implement a full SCPI
        # parsing here

        if cmd[-1] == '?':
            if cmd in self.state.keys():
                return self.state[cmd]
        else:
            (par, val) = cmd.split(' ')
            self.state[par] = val
