"""
This module defines the dict representation of the ``RunDescriber`` used for
serialization and deserialization of the ``RunDescriber``.

RunDescriber version log:

- 0: The run_describer has a single attribute, interdependencies, which is an
instance of InterDependencies (which contains ParamSpecs)
- 1: The run_describer has a single attribute, interdependencies, which is an
instance of InterDependencies_ (which contains ParamSpecBases)
- 2: The run_describer has a two attribute, interdependencies, which is an
instance of InterDependencies (which contains ParamSpecs) and
interdependencies_, which is an instance of InterDependencies_
(which contains ParamSpecBases)
"""

from typing import Dict, Any, List, Tuple, Union
from typing_extensions import TypedDict

from ..param_spec import ParamSpecDict


class InterDependenciesDict(TypedDict):
    paramspecs: Tuple[ParamSpecDict, ...]


class InterDependencies_Dict(TypedDict):
    parameters: Dict[str, Any]
    dependencies: Dict[str, Any]
    inferences: Dict[str, Any]
    standalones: List[Any]


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

RunDescriberDicts = Union[RunDescriberV0Dict, RunDescriberV1Dict, RunDescriberV2Dict]
