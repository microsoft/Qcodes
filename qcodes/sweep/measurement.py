from qcodes.dataset.measurements import Measurement
from qcodes.sweep.base import BaseSweepObject


class SweepMeasurement(Measurement):
    def register_sweep(self, sweep_object: BaseSweepObject) -> None:

        sweep_object.parameter_table.resolve_dependencies()
        param_specs = sweep_object.parameter_table.param_specs

        self.parameters = {
            p.name: p for p in param_specs
        }
