import pytest
from qcodes import ParamSpec
from qcodes.sweep.base import ParamTable


def test_nest():
    """
    Test the simple 1D nesting functionality.
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])

    # This represents a 1D sweep such as:
    # nest(sweep(a, [0, 1, 2]), b)
    # that is, sweep a and measure b at each set point
    table_nest = table_a.nest(table_b)
    table_nest.resolve_dependencies()

    # Extract specs from the table
    table_specs = table_nest.param_specs
    assert len(table_specs) == 2
    # The specs extracted should have the correct dependencies
    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == 'a'
    # The original specs should not be touched
    assert b.depends_on == ''


def test_double_nest():
    """
    Test the 2D nesting functionality
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])
    table_c = ParamTable([c])

    # A 2D sweep
    # table_nest = table_a.nest(table_b.nest(table_c))
    table_nest = table_a.nest(table_b).nest(table_c)
    table_nest.resolve_dependencies()

    # Extract specs from the table
    table_specs = table_nest.param_specs
    table_specs = sorted(table_specs, key=lambda v: v.name)
    assert len(table_specs) == 3

    # The specs extracted should have the correct dependencies
    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == ''

    assert table_specs[2].name == 'c'
    assert table_specs[2].depends_on == 'a, b'
    # The original specs should not be touched
    assert a.depends_on == ''
    assert b.depends_on == ''
    assert c.depends_on == ''


def test_inferred_from():
    """
    Test the 2D nesting functionality whereby the measurement generates two
    parameters, the second being inferred from the first
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")
    d = ParamSpec("d", paramtype="numeric", inferred_from='c')

    table_a = ParamTable([a])
    table_b = ParamTable([b])
    table_c = ParamTable([c, d])

    table_nest = table_a.nest(table_b).nest(table_c)
    table_nest.resolve_dependencies()

    # Extract specs from the table
    table_specs = table_nest.param_specs
    table_specs = sorted(table_specs, key=lambda v: v.name)
    assert len(table_specs) == 4

    # The specs extracted should have the correct dependencies
    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == ''

    assert table_specs[2].name == 'c'
    assert table_specs[2].depends_on == 'a, b'

    assert table_specs[3].name == 'd'
    assert table_specs[3].depends_on == 'a, b'
    assert table_specs[3].inferred_from == 'c'

    # The original specs should not be touched
    assert a.depends_on == ''
    assert b.depends_on == ''
    assert c.depends_on == ''


def test_not_allowed():
    """
    We can make sweep objects which produce contradictory dependencies.
    In this test, the table contains a contradiction whereby 'c' is
    dependent on both 'only a' and 'only b'.
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")

    table_a = ParamTable([a, b])
    table_c = ParamTable([c])

    with pytest.raises(ValueError):
        table_a.nest(table_c)


def test_chain():
    """
    Test simple chaining
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])

    table_chain = table_a.chain(table_b)
    table_chain.resolve_dependencies()

    # Extract specs from the table
    table_specs = table_chain.param_specs
    assert len(table_specs) == 2

    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == ''


def test_nest_chain():
    """
    We test the following in pseudo-code

    for a in [0, 1, 2]:
        b()
        c()

    Both 'b' and 'c' depend on 'a'
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])
    table_c = ParamTable([c])

    table_result = table_a.nest(table_b.chain(table_c))
    table_result.resolve_dependencies()

    # Extract specs from the table
    table_specs = table_result.param_specs
    assert len(table_specs) == 3

    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == 'a'

    assert table_specs[2].name == 'c'
    assert table_specs[2].depends_on == 'a'


def test_nest_chain_chain():
    """
    We test the following in pseudo-code

    for a in [0, 1, 2]:
        b()
        c()
        d()

    Both 'b', 'c' and 'd' depend on 'a'
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")
    d = ParamSpec("d", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])
    table_c = ParamTable([c])
    table_d = ParamTable([d])

    table_result = table_a.nest(table_b.chain(table_c).chain(table_d))
    table_result.resolve_dependencies()

    # Extract specs from the table
    table_specs = table_result.param_specs
    assert len(table_specs) == 4

    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == 'a'

    assert table_specs[2].name == 'c'
    assert table_specs[2].depends_on == 'a'

    assert table_specs[3].name == 'd'
    assert table_specs[3].depends_on == 'a'


def test_not_allowed_2():
    """
    We can make sweep objects which produce contradictory dependencies.
    In this test, the table contains a contradiction whereby 'c' is
    dependent on both 'only a' and 'only b'. We use chaining to make the
    contradiction
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])
    table_c = ParamTable([c])

    with pytest.raises(ValueError):
        table_a.chain(table_b).nest(table_c)


def test_nest_chain_nest():
    """
    Test a sweep which is equivalent to

    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            c()
        for d in [0, 1, 2]:
            e()
    """
    a = ParamSpec("a", paramtype="numeric")
    b = ParamSpec("b", paramtype="numeric")
    c = ParamSpec("c", paramtype="numeric")
    d = ParamSpec("d", paramtype="numeric")
    e = ParamSpec("e", paramtype="numeric")

    table_a = ParamTable([a])
    table_b = ParamTable([b])
    table_c = ParamTable([c])
    table_d = ParamTable([d])
    table_e = ParamTable([e])

    table_result = table_a.nest(
        table_b.nest(table_c).chain(table_d.nest(table_e))
    )

    table_result.resolve_dependencies()
    table_specs = table_result.param_specs
    assert len(table_specs) == 5

    assert table_specs[0].name == 'a'
    assert table_specs[0].depends_on == ''

    assert table_specs[1].name == 'b'
    assert table_specs[1].depends_on == ''

    assert table_specs[2].name == 'c'
    assert table_specs[2].depends_on == 'a, b'

    assert table_specs[3].name == 'd'
    assert table_specs[3].depends_on == ''

    assert table_specs[4].name == 'e'
    assert table_specs[4].depends_on == 'a, d'
