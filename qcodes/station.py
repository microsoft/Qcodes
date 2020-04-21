"""Station objects - collect all the equipment you use to do an experiment."""
from singleton_decorator import singleton

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, DelegateAttributes

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import _BaseParameter
from qcodes.instrument.parameter_node import ParameterNode

from qcodes.actions import _actions_snapshot


@singleton
class Station(ParameterNode):

    """
    A representation of the entire physical setup.

    Lists all the connected `Component`\s and the current default
    measurement (a list of actions). Contains a convenience method
    `.measure()` to measure these defaults right now, but this is separate
    from the code used by `Loop`.

    Args:
        *components (list[Any]): components to add immediately to the Station.
            can be added later via self.add_component

        update_snapshot (bool): immediately update the snapshot
            of each component as it is added to the Station, default true

    Attributes:
        delegate_attr_dicts (list): a list of names (strings) of dictionaries which are
            (or will be) attributes of self, whose keys should be treated as
            attributes of self
    """

    default = None

    def __init__(self, *components, update_snapshot=True, **kwargs):
        super().__init__(use_as_attributes=True, **kwargs)

        self.components = {}
        for item in components:
            self.add_component(item, update_snapshot=update_snapshot)

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
            "instruments": {},
            "parameters": {},
            "components": {},
            "default_measurement": _actions_snapshot(self.default_measurement, update),
        }

        for name, item in self.components.items():
            if isinstance(item, (Instrument)):
                snap["instruments"][name] = item.snapshot(update=update)
            elif isinstance(item, _BaseParameter):
                snap["parameters"][name] = item.snapshot(update=update)
            else:
                snap["components"][name] = item.snapshot(update=update)

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
            name = getattr(component, "name", f"component{len(self.components)}")
        name = make_unique(str(name), self.components)

        self.components[name] = component

        return name

    # station['someitem'] and station.someitem are both
    # shortcuts to station.components['someitem']
    # (assuming 'someitem' doesn't have another meaning in Station)
    def __getitem__(self, key):
        """Shortcut to components dict."""
        return self.components[key]

    def __contains__(self, key):
        return key in self.components

    delegate_attr_dicts = ["components"]
