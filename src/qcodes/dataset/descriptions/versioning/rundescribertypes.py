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

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..param_spec import ParamSpecBaseDict, ParamSpecDict


class InterDependenciesDict(TypedDict):
    paramspecs: tuple[ParamSpecDict, ...]


class InterDependencies_Dict(TypedDict):
    parameters: dict[str, ParamSpecBaseDict]
    dependencies: dict[str, list[str]]
    inferences: dict[str, list[str]]
    standalones: list[str]


Shapes = dict[str, tuple[int, ...]]


class RunDescriberV0Dict(TypedDict):
    version: int
    interdependencies: InterDependenciesDict


class RunDescriberV1Dict(TypedDict):
    version: int
    interdependencies: InterDependencies_Dict


class RunDescriberV2Dict(RunDescriberV0Dict):
    interdependencies_: InterDependencies_Dict


class RunDescriberV3Dict(RunDescriberV2Dict):
    shapes: Shapes | None
    # dict from dependent to dict from dependency to num points in grid


RunDescriberDicts = (
    RunDescriberV0Dict | RunDescriberV1Dict | RunDescriberV2Dict | RunDescriberV3Dict
)
