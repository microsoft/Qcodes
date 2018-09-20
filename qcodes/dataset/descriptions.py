from ruamel.yaml import YAML

from qcodes.dataset.dependencies import InterDependencies


class RunDescriber:

    def __init__(self, interdeps: InterDependencies) -> None:
        self.interdeps = interdeps

    def output_yaml(self):
        pass
