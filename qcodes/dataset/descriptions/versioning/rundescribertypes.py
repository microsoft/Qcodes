"""
This module defines the dict representation of the ``RunDescriber`` used for
serialization and deserialization of the ``RunDescriber``.

RunDescriber version log:

- 0: The run_describer has a single attribute, interdependencies, which is an
instance of InterDependencies (which contains ParamSpecs)
- 1: The run_describer has a single attribute, interdependencies, which is an
instance of InterDependencies_ (which contains ParamSpecBases)
- 2: The run_describer has a two attributes: interdependencies, which is an
instance of InterDependencies (which contains ParamSpecs) and
interdependencies_, which is an instance of InterDependencies_
(which contains ParamSpecBases)
"""

from typing import Dict, List, Tuple, Union

from typing_extensions import TypedDict

from ..param_spec import ParamSpecBaseDict, ParamSpecDict


class InterDependenciesDict(TypedDict):
    paramspecs: Tuple[ParamSpecDict, ...]


class InterDependencies_Dict(TypedDict):
    parameters: Dict[str, ParamSpecBaseDict]
    dependencies: Dict[str, List[str]]
    inferences: Dict[str, List[str]]
    standalones: List[str]


class RunDescriberV0Dict(TypedDict):
    version: int
    interdependencies: "InterDependenciesDict"


class RunDescriberV1Dict(TypedDict):
    version: int
    interdependencies: "InterDependencies_Dict"


class RunDescriberV2Dict(TypedDict):
    version: int
    interdependencies: "InterDependenciesDict"
    interdependencies_: "InterDependencies_Dict"

RunDescriberDicts = Union[RunDescriberV0Dict,
                          RunDescriberV1Dict,
                          RunDescriberV2Dict]
