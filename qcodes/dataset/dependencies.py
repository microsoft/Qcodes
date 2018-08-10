import io

from ruamel.yaml import YAML

from qcodes.dataset.param_spec import ParamSpec


class InterDependencies:
    """
    An object holding the same information as the yaml file plus
    methods for validation and extraction of data

    My idea is to have the helper functions for plotting (get_layout, get_XX)
    not call the SQLite database but this object instead. That is to say,
    the dependencies text will be read out, but all further processing happens
    via this object
    """

    def __init__(self, *paramspecs: ParamSpec) -> None:
        self.paramspecs = paramspecs


def interdeps_to_yaml(idp: InterDependencies) -> str:
    """
    Output the dependencies as a yaml string
    """
    yaml = YAML()
    yaml.register_class(ParamSpec)
    stream = io.StringIO()
    # we use a heading, Parameters, since the yaml file might be extended
    # with more info in the future
    yaml.dump({'Parameters': idp.paramspecs}, stream=stream)
    output = stream.getvalue()
    stream.close()
    return output


def yaml_to_interdeps(yaml_str: str) -> InterDependencies:
    yaml = YAML()
    yaml.register_class(ParamSpec)
    paramspecs = yaml.load(yaml_str)['Parameters']
    return InterDependencies(paramspecs)

