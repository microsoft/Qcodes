from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, safe_getattr


class Station(Metadatable):
    '''
    A representation of the entire physical setup.

    Lists all the connected `Instrument`s and the current default
    measurement (a list of actions). Contains a convenience method
    `.measure()` to measure these defaults right now, but this is separate
    from the code used by `Loop`.
    '''
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
        '''
        Record one instrument as part of this Station

        Returns the name assigned this instrument, which may have
        been changed to make it unique among previously added instruments.
        '''
        if name is None:
            name = getattr(instrument, 'name',
                           'instrument{}'.format(len(self.instruments)))
        name = make_unique(str(name), self.instruments)
        self.instruments[name] = instrument
        return name

    def set_measurement(self, *actions):
        '''
        Save a set *actions as the default measurement for this Station

        These actions will be executed by default by a Loop if this is the
        default Station, and any measurements among them can be done once
        by .measure
        '''
        self.default_measurement = actions

    def measure(self, *actions):
        '''
        Measure any parameters in *actions, or the default measurement
        for this station if none are provided.
        '''
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
