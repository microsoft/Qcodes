"""
This module contains functions which implement conversion between different
(neighbouring) versions of RunDescriber.



"""

from __future__ import annotations

from ..dependencies import InterDependencies_
from ..param_spec import ParamSpec, ParamSpecBase
from .rundescribertypes import (
    RunDescriberV0Dict,
    RunDescriberV1Dict,
    RunDescriberV2Dict,
    RunDescriberV3Dict,
)
from .v0 import InterDependencies


def old_to_new(idps: InterDependencies) -> InterDependencies_:
    """
    Create a new InterDependencies_ object (new style) from an existing
    InterDependencies object (old style). Leaves the original object unchanged.
    Incidentally, this function can serve as a validator of the original object
    """
    namedict: dict[str, ParamSpec] = {ps.name: ps for ps in idps.paramspecs}

    dependencies = {}
    inferences = {}
    standalones_mut = []
    root_paramspecs: list[ParamSpecBase] = []

    for ps in idps.paramspecs:
        deps = tuple(namedict[n].base_version() for n in ps.depends_on_)
        inffs = tuple(namedict[n].base_version() for n in ps.inferred_from_)
        if len(deps) > 0:
            dependencies.update({ps.base_version(): deps})
            root_paramspecs += list(deps)
        if len(inffs) > 0:
            inferences.update({ps.base_version(): inffs})
            root_paramspecs += list(inffs)
        if len(deps) == len(inffs) == 0:
            standalones_mut.append(ps.base_version())

    standalones = tuple(set(standalones_mut).difference(set(root_paramspecs)))

    idps_ = InterDependencies_(
        dependencies=dependencies, inferences=inferences, standalones=standalones
    )
    return idps_


def new_to_old(idps: InterDependencies_) -> InterDependencies:
    """
    Create a new InterDependencies object (old style) from an existing
    InterDependencies_ object (new style). Leaves the original object
    unchanged. Only meant to be used for ensuring backwards-compatibility
    until we update sqlite module to forget about ParamSpecs
    """

    paramspecs: dict[str, ParamSpec] = {}

    # first the independent parameters
    for indeps in idps.dependencies.values():
        for indep in indeps:
            paramspecs.update(
                {
                    indep.name: ParamSpec(
                        name=indep.name,
                        paramtype=indep.type,
                        label=indep.label,
                        unit=indep.unit,
                    )
                }
            )

    for inffs in idps.inferences.values():
        for inff in inffs:
            paramspecs.update(
                {
                    inff.name: ParamSpec(
                        name=inff.name,
                        paramtype=inff.type,
                        label=inff.label,
                        unit=inff.unit,
                    )
                }
            )

    for ps_base in idps._paramspec_to_id.keys():
        paramspecs.update(
            {
                ps_base.name: ParamSpec(
                    name=ps_base.name,
                    paramtype=ps_base.type,
                    label=ps_base.label,
                    unit=ps_base.unit,
                )
            }
        )

    for ps, indeps in idps.dependencies.items():
        for indep in indeps:
            paramspecs[ps.name]._depends_on.append(indep.name)
    for ps, inffs in idps.inferences.items():
        for inff in inffs:
            paramspecs[ps.name]._inferred_from.append(inff.name)

    return InterDependencies(*tuple(paramspecs.values()))


def v0_to_v1(old: RunDescriberV0Dict) -> RunDescriberV1Dict:
    """
    Convert a v0 RunDescriber Dict to a v1 RunDescriber Dict
    """
    old_idps = InterDependencies._from_dict(old["interdependencies"])
    new_idps_dict = old_to_new(old_idps)._to_dict()
    return RunDescriberV1Dict(version=1, interdependencies=new_idps_dict)


def v1_to_v2(old: RunDescriberV1Dict) -> RunDescriberV2Dict:
    """
    Convert a v1 RunDescriber Dict to a v2 RunDescriber Dict
    """
    interdeps_dict = old["interdependencies"]
    interdeps_ = InterDependencies_._from_dict(interdeps_dict)
    interdepsdict = new_to_old(interdeps_)._to_dict()
    return RunDescriberV2Dict(
        version=2, interdependencies_=interdeps_dict, interdependencies=interdepsdict
    )


def v2_to_v3(old: RunDescriberV2Dict) -> RunDescriberV3Dict:
    return RunDescriberV3Dict(
        version=3,
        interdependencies=old["interdependencies"],
        interdependencies_=old["interdependencies_"],
        shapes=None,
    )


def v0_to_v2(old: RunDescriberV0Dict) -> RunDescriberV2Dict:
    """
    Convert a v0 RunDescriber Dict to a v2 RunDescriber Dict
    """
    return v1_to_v2(v0_to_v1(old))


def v0_to_v3(old: RunDescriberV0Dict) -> RunDescriberV3Dict:
    return v2_to_v3(v0_to_v2(old))


def v1_to_v3(old: RunDescriberV1Dict) -> RunDescriberV3Dict:
    return v2_to_v3(v1_to_v2(old))


def v3_to_v2(new: RunDescriberV3Dict) -> RunDescriberV2Dict:
    return RunDescriberV2Dict(
        version=2,
        interdependencies=new["interdependencies"],
        interdependencies_=new["interdependencies_"],
    )


def v2_to_v1(new: RunDescriberV2Dict) -> RunDescriberV1Dict:
    """
    Convert a v2 RunDescriber Dict to a v1 RunDescriber Dict
    """
    rundescriberdictv1 = RunDescriberV1Dict(
        version=1, interdependencies=new["interdependencies_"]
    )
    return rundescriberdictv1


def v1_to_v0(new: RunDescriberV1Dict) -> RunDescriberV0Dict:
    """
    Convert a v1 RunDescriber Dict to a v0 RunDescriber Dict
    """
    interdeps_dict = new["interdependencies"]
    interdeps_ = InterDependencies_._from_dict(interdeps_dict)
    interdepsdict = new_to_old(interdeps_)._to_dict()
    rundescriberv0dict = RunDescriberV0Dict(version=0, interdependencies=interdepsdict)
    return rundescriberv0dict


def v3_to_v1(new: RunDescriberV3Dict) -> RunDescriberV1Dict:
    return v2_to_v1(v3_to_v2(new))


def v2_to_v0(new: RunDescriberV2Dict) -> RunDescriberV0Dict:
    """
    Convert a v2 RunDescriber Dict to a v0 RunDescriber Dict
    """
    return v1_to_v0(v2_to_v1(new))


def v3_to_v0(new: RunDescriberV3Dict) -> RunDescriberV0Dict:
    return v1_to_v0(v3_to_v1(new))
