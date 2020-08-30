from typing import Dict, Any, List, Tuple, Union
from typing_extensions import TypedDict


class InterDependenciesDict(TypedDict):
    paramspecs: Tuple[Dict[str, Any], ...]


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
