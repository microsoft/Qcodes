from qcodes.instrument.base import Instrument
import requests


class HTTPInstrument(Instrument):

    def __init__(self, name, address, port, timeout=5,
                 terminator='\n', persistent=True, write_confirmation=True,
                 **kwargs):
        super().__init__(name, **kwargs)


        self._address = address
        self._port = port
        self._timeout = timeout
        self._terminator = terminator
        self._confirmation = write_confirmation
        self._session = requests.Session()


    def close(self):
        self._session.close()
        super().close()

    def ask_raw(self, cmd):
        result = self._session.get('http://localhost:8000/{}'.format(cmd)).text
        return result

    def write_raw(self, cmd):
        response = self._session.post('http://localhost:8000/{}'.format(cmd))

    def set_with_post(self, parameter, value):
        data = {"setpoint": value}
        response = self._session.post('http://localhost:8000/{}'.format(parameter), json=data)
