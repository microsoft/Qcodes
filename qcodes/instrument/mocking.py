from qcodes.instrument.mockers.base import FakeSCPIInstrument


class Nothing:
    """
    This class is currently required by InstrumentBase to exist,
    and to take a name as an argument to __init__
    """
    def __init__(self, name: str) -> None:
        pass


class MockSCPIInstrumentDriver:
    """
    Class that can tweak standard SCPI-speaking instrument driver to
    work with a mock instrument.  In this mocking framework, we
    distinguish between the instrument and its driver. We mock the
    actual instrument and hack/configure the driver slightly to
    communicate with the mock instrument instead of a real instrument.

    Args:
        name: The name that the mock driver instance will have
        driver_to_test: The driver we'd like to work with, a
            QCoDeS Instrument class (not instance!)
        fakeinstrument: An instance of the mock of the actual instrument
    """

    def __init__(self, name: str, driver_to_test: type,
                 fakeinstrument: FakeSCPIInstrument) -> None:

        # we assign Nothing because the InstrumentBase class needs it
        driver_to_test.mocker_class = Nothing

        # TODO: Perhaps find a better name than using __call__
        driver_to_test.write = fakeinstrument.__call__
        driver_to_test.ask = fakeinstrument.__call__

        self.driver = driver_to_test(name, address=None,
                                     testing=True)

        self.initialise_params()

    def initialise_params(self):

        parents = [self.driver] + list(self.driver.submodules.values())

        for parent in parents:
            for param in parent.parameters.values():
                if param.vals:
                    val = param.vals.valid_values[0]
                    if hasattr(param, 'set'):
                        param.set(val)
