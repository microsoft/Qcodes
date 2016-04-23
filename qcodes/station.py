from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, DelegateAttributes

from qcodes.instrument.remote import RemoteInstrument
from qcodes.instrument.remote import RemoteParameter
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import StandardParameter

class Station(Metadatable, DelegateAttributes):
    '''
    A representation of the entire physical setup.

    Lists all the connected `Item`s and the current default
    measurement (a list of actions). Contains a convenience method
    `.measure()` to measure these defaults right now, but this is separate
    from the code used by `Loop`.
    '''
    default = None

    def __init__(self, *items, monitor=None, default=True, **kwargs):
        super().__init__(**kwargs)

        # when a new station is defined, store it in a class variable
        # so it becomes the globally accessible default station.
        # You can still have multiple stations defined, but to use
        # other than the default one you must specify it explicitly.
        # If for some reason you want this new Station NOT to be the
        # default, just specify default=False
        if default:
            Station.default = self

        self.items = {}
        for item in items:
            self.add_item(item)

        self.monitor = monitor

    def snapshot_base(self, *args, **kwargs):
        snap = {'instruments': {},
                'parameters': {},
                'items': {},
                'station': {}}

        snap['station'] = {}

        for name, ins in self.items.items():
            if isinstance(ins, (RemoteInstrument,
                                Instrument)):
                snap['instruments'][name] = ins.snapshot(*args, **kwargs)
            elif isinstance(ins, (StandardParameter,
                                  RemoteParameter)):
                snap['parameters'][name] = ins.snapshot(*args, **kwargs)
            else:
                snap['items'][name] = ins.snapshot(*args, **kwargs)

        return snap

    def add_item(self, item, name=None):
        '''
        Record one item as part of this Station

        Returns the name assigned this item, which may have
        been changed to make it unique among previously added items.
        '''
        if name is None:
            name = getattr(item, 'name',
                           'item{}'.format(len(self.items)))
        name = make_unique(str(name), self.items)
        self.items[name] = item
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

    # station['someitem'] and station.someitem are both
    # shortcuts to station.items['someitem']
    # (assuming 'someitem' doesn't have another meaning in Station)
    def __getitem__(self, key):
        return self.items[key]

    delegate_attr_dicts = ['items']
