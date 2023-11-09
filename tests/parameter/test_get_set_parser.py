from qcodes.parameters import Parameter

from .conftest import ParameterMemory


def test_param_cmd_with_parsing() -> None:

    mem = ParameterMemory()

    p = Parameter('p_int', get_cmd=mem.get, get_parser=int,
                  set_cmd=mem.set, set_parser=mem.parse_set_p)

    p(5)
    assert mem.get() == '5'
    assert p() == 5

    p.cache.set(7)
    assert p.get_latest() == 7
    # Nothing has been passed to the "instrument" at ``cache.set``
    # call, hence the following assertions should hold
    assert mem.get() == '5'
    assert p() == 5
    assert p.get_latest() == 5
