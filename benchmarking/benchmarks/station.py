"""
Benchmarks for the cost of snapshotting a station with many nested
instrument modules and parameters. This is dominated by pure Python
work, such as resolving the names of parameters, rather than by
communication with instruments.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Any, ClassVar

import qcodes
from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.database import initialise_database
from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBase,
    InstrumentChannel,
    InstrumentModule,
)
from qcodes.parameters import Parameter
from qcodes.station import Station


def _add_parameters(module: InstrumentBase, n_parameters: int) -> None:
    for i in range(n_parameters):
        module.add_parameter(
            f"param{i}",
            parameter_class=Parameter,
            get_cmd=None,
            set_cmd=None,
            initial_value=i,
            unit="V",
        )


class _Leaf(InstrumentModule):
    def __init__(
        self, parent: InstrumentBase, name: str, n_parameters: int, **kwargs: Any
    ) -> None:
        super().__init__(parent, name, **kwargs)
        _add_parameters(self, n_parameters)


class _Channel(InstrumentChannel):
    def __init__(
        self, parent: InstrumentBase, name: str, n_parameters: int, **kwargs: Any
    ) -> None:
        super().__init__(parent, name, **kwargs)
        _add_parameters(self, n_parameters)
        self.add_submodule("leaf", _Leaf(self, "leaf", n_parameters))


class _Module(InstrumentModule):
    def __init__(
        self,
        parent: InstrumentBase,
        name: str,
        n_channels: int,
        n_parameters: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, name, **kwargs)
        _add_parameters(self, n_parameters)
        channels = ChannelList(self, "channels", _Channel)
        for i in range(n_channels):
            channel = _Channel(self, f"ch{i}", n_parameters)
            channels.append(channel)
            self.add_submodule(f"ch{i}", channel)
        self.add_submodule("channels", channels.to_channel_tuple())


class NestedDummyInstrument(Instrument):
    """
    A dummy instrument with the structure
    ``instrument -> module -> channel -> leaf -> parameter``.
    Every level holds ``n_parameters`` parameters.
    """

    def __init__(
        self,
        name: str,
        n_modules: int,
        n_channels: int,
        n_parameters: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        _add_parameters(self, n_parameters)
        for i in range(n_modules):
            self.add_submodule(
                f"module{i}", _Module(self, f"module{i}", n_channels, n_parameters)
            )

    def get_idn(self) -> dict[str, str | None]:
        return {"vendor": "QCoDeS", "model": type(self).__name__}


# 4 modules * 8 channels * 2 levels * 30 parameters + module and root
# parameters give a bit more than 2000 parameters per instrument
_INSTRUMENT_KWARGS: dict[str, int] = {
    "n_modules": 4,
    "n_channels": 8,
    "n_parameters": 30,
}


def _make_station(n_instruments: int) -> Station:
    station = Station()
    for i in range(n_instruments):
        station.add_component(NestedDummyInstrument(f"dummy{i}", **_INSTRUMENT_KWARGS))
    return station


class StationSnapshot:
    """
    Time taking a snapshot of a station without updating parameters.
    """

    params: ClassVar[list[int]] = [1, 2]
    param_names: ClassVar[list[str]] = ["n_instruments"]
    timer = time.perf_counter

    def setup(self, n_instruments: int) -> None:
        self.station = _make_station(n_instruments)

    def teardown(self, n_instruments: int) -> None:
        Instrument.close_all()
        Station.default = None

    def time_snapshot(self, n_instruments: int) -> None:
        self.station.snapshot(update=False)


class MeasurementRunEnterExit:
    """
    Time entering and exiting the ``Measurement.run`` context manager,
    which takes a snapshot of the default station at the start of the run.
    """

    params: ClassVar[list[int]] = [1, 2]
    param_names: ClassVar[list[str]] = ["n_instruments"]
    timer = time.perf_counter

    def setup(self, n_instruments: int) -> None:
        self.tmpdir = tempfile.mkdtemp()
        qcodes.config["core"]["db_location"] = os.path.join(self.tmpdir, "temp.db")
        qcodes.config["core"]["db_debug"] = False
        initialise_database()
        self.experiment = new_experiment("test-experiment", sample_name="test-sample")

        self.station = _make_station(n_instruments)
        instrument = self.station.components["dummy0"]
        assert isinstance(instrument, NestedDummyInstrument)
        setpoint = instrument.module0.ch0.leaf.param0
        measured = instrument.module0.ch0.leaf.param1

        self.meas = Measurement(self.experiment, station=self.station)
        self.meas.register_parameter(setpoint)
        self.meas.register_parameter(measured, setpoints=(setpoint,))

    def teardown(self, n_instruments: int) -> None:
        Instrument.close_all()
        Station.default = None
        self.experiment.conn.close()
        shutil.rmtree(self.tmpdir)

    def time_run(self, n_instruments: int) -> None:
        with self.meas.run():
            pass
