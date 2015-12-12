from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, safe_getattr


class Station(Metadatable):
    default = None

    def __init__(self, *instruments, monitor=None, default=True):
        # when a new station is defined, store it in a class variable
        # so it becomes the globally accessible default station.
        # You can still have multiple stations defined, but to use
        # other than the default one you must specify it explicitly.
        # If for some reason you want this new Station NOT to be the
        # default, just specify default=False
        if default:
            Station.default = self

        self.instruments = {}
        for instrument in instruments:
            self.add_instrument(instrument)

        self.monitor = monitor

    def add_instrument(self, instrument, name=None):
        if name is None:
            name = getattr(instrument, 'name',
                           'instrument{}'.format(len(self.instruments)))
        name = make_unique(str(name), self.instruments)
        self.instruments[name] = instrument
        return name

    def set_measurement(self, *actions):
        self.default_measurement = actions

    def measure(self, *actions):
        if not actions:
            actions = self.default_measurement

        out = []

        # this is a stripped down, uncompiled version of how
        # ActiveLoop handles a set of actions
        # callables (including Wait) return nothing, but can
        # change system state.
        for action in actions:
            if hasattr(action, 'get'):
                out.append(action.get())
            elif callable(action):
                action()

        return out

    # station['someinstrument'] and station.someinstrument are both
    # shortcuts to station.instruments['someinstrument']
    # (assuming 'someinstrument' doesn't have another meaning in Station)
    def __getitem__(self, key):
        return self.instruments[key]

    def __getattr__(self, key):
        return safe_getattr(self, key, 'instruments')
