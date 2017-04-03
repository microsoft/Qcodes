"""Station objects - collect all the equipment you use to do an experiment."""

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, DelegateAttributes

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import StandardParameter

from qcodes.actions import _actions_snapshot


class Station(Metadatable, DelegateAttributes):

    """
    A representation of the entire physical setup.

    Lists all the connected `Component`\s and the current default
    measurement (a list of actions). Contains a convenience method
    `.measure()` to measure these defaults right now, but this is separate
    from the code used by `Loop`.

    Args:
        *components (list[Any]): components to add immediately to the Station.
            can be added later via self.add_component

        monitor (None): Not implememnted, the object that monitors the system continuously

        default (bool): is this station the default, which gets
            used in Loops and elsewhere that a Station can be specified, default  true

        update_snapshot (bool): immediately update the snapshot
            of each component as it is added to the Station, default true

    Attributes:
        default (Station): class attribute to store the default station
        delegate_attr_dicts (list): a list of names (strings) of dictionaries which are
            (or will be) attributes of self, whose keys should be treated as
            attributes of self
    """

    default = None

    def __init__(self, *components, monitor=None, default=True,
                 update_snapshot=True, **kwargs):
        super().__init__(**kwargs)

        # when a new station is defined, store it in a class variable
        # so it becomes the globally accessible default station.
        # You can still have multiple stations defined, but to use
        # other than the default one you must specify it explicitly.
        # If for some reason you want this new Station NOT to be the
        # default, just specify default=False
        if default:
            Station.default = self

        self.components = {}
        for item in components:
            self.add_component(item, update_snapshot=update_snapshot)

        self.monitor = monitor

        self.default_measurement = []

    def snapshot_base(self, update=False):
        """
        State of the station as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
             all the childs: f.ex. instruments, parameters, components, etc.
             If False, just use the latest values in memory.

        Returns:
            dict: base snapshot
        """
        snap = {
            'instruments': {},
            'parameters': {},
            'components': {},
            'default_measurement': _actions_snapshot(
                self.default_measurement, update)
        }

        for name, itm in self.components.items():
            if isinstance(itm, (Instrument)):
                snap['instruments'][name] = itm.snapshot(update=update)
            elif isinstance(itm, (Parameter,
                                  ManualParameter,
                                  StandardParameter
                                  )):
                snap['parameters'][name] = itm.snapshot(update=update)
            else:
                snap['components'][name] = itm.snapshot(update=update)

        return snap

    def add_component(self, component, name=None, update_snapshot=True):
        """
        Record one component as part of this Station.

        Args:
            component (Any): components to add to the Station.
            name (str): name of the component
            update_snapshot (bool): immediately update the snapshot
                of each component as it is added to the Station, default true

        Returns:
            str: The name assigned this component, which may have been changed to
            make it unique among previously added components.

        """
        try:
            component.snapshot(update=update_snapshot)
        except:
            pass
        if name is None:
            name = getattr(component, 'name',
                           'component{}'.format(len(self.components)))
        name = make_unique(str(name), self.components)
        self.components[name] = component
        return name

    def set_measurement(self, *actions):
        """
        Save a set \*actions as the default measurement for this Station.

        These actions will be executed by default by a Loop if this is the
        default Station, and any measurements among them can be done once
        by .measure
        Args:
            *actions: parameters to set as default  measurement
        """
        # Validate now so the user gets an error message ASAP
        # and so we don't accept `Loop` as an action here, where
        # it would cause infinite recursion.
        # We need to import Loop inside here to avoid circular import
        from .loops import Loop
        Loop.validate_actions(*actions)

        self.default_measurement = actions

    def measure(self, *actions):
        """
        Measure the default measurement, or parameters in actions.

        Args:
            *actions: parameters to mesure
        """
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
    # shortcuts to station.components['someitem']
    # (assuming 'someitem' doesn't have another meaning in Station)
    def __getitem__(self, key):
        """Shortcut to components dict."""
        return self.components[key]

    delegate_attr_dicts = ['components']
