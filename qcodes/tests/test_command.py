import pytest

from qcodes.parameters.command import Command, NoCommandError


class CustomError(Exception):
    pass


def test_bad_calls():
    with pytest.raises(TypeError):
        Command()

    with pytest.raises(TypeError):
        Command(cmd='')

    with pytest.raises(TypeError):
        Command(0, '', output_parser=lambda: 1)

    with pytest.raises(TypeError):
        Command(1, '', input_parser=lambda: 1)

    with pytest.raises(TypeError):
        Command(0, cmd='', exec_str='not a function')

    with pytest.raises(TypeError):
        Command(0, cmd=lambda: 1, no_cmd_function='not a function')


def test_no_cmd():
    with pytest.raises(NoCommandError):
        Command(0)

    def no_cmd_function():
        raise CustomError('no command')

    no_cmd = Command(0, no_cmd_function=no_cmd_function)
    with pytest.raises(CustomError):
        no_cmd()


def test_cmd_str():
    def f_now(x):
        return x + ' now'

    def upper(s):
        return s.upper()

    def reversestr(s):
        return s[::-1]

    def swap(a, b):
        return b, a

    # basic exec_str
    cmd = Command(0, 'pickles', exec_str=f_now)
    assert cmd() == 'pickles now'

    # with output parsing
    cmd = Command(0, 'blue', exec_str=f_now, output_parser=upper)
    assert cmd() == 'BLUE NOW'

    # parameter insertion
    cmd = Command(3, '{} is {:.2f}% better than {}', exec_str=f_now)
    assert cmd('ice cream', 56.2, 'cake') == \
                     'ice cream is 56.20% better than cake now'
    with pytest.raises(ValueError):
        cmd('cake', 'a whole lot', 'pie')

    with pytest.raises(TypeError):
        cmd('donuts', 100, 'bagels', 'with cream cheese')

    # input parsing
    cmd = Command(1, 'eat some {}', exec_str=f_now, input_parser=upper)
    assert cmd('ice cream') == 'eat some ICE CREAM now'

    # input *and* output parsing
    cmd = Command(1, 'eat some {}', exec_str=f_now,
                  input_parser=upper, output_parser=reversestr)
    assert cmd('ice cream') == 'won MAERC ECI emos tae'

    # multi-input parsing, no output parsing
    cmd = Command(2, '{} and {}', exec_str=f_now, input_parser=swap)
    assert cmd('I', 'you') == 'you and I now'

    # multi-input parsing *and* output parsing
    cmd = Command(2, '{} and {}', exec_str=f_now,
                  input_parser=swap, output_parser=upper)
    assert cmd('I', 'you') == 'YOU AND I NOW'


def test_cmd_function():
    def myexp(a, b):
        return a ** b

    cmd = Command(2, myexp)
    assert cmd(10, 3) == 1000

    with pytest.raises(TypeError):
        Command(3, myexp)

    # with output parsing
    cmd = Command(2, myexp, output_parser=lambda x: 5 * x)
    assert cmd(10, 3) == 5000

    # input parsing
    cmd = Command(1, abs, input_parser=lambda x: x + 1)
    assert cmd(-10) == 9

    # input *and* output parsing
    cmd = Command(1, abs, input_parser=lambda x: x + 2,
                  output_parser=lambda x: 3 * x)
    assert cmd(-6) == 12

    # multi-input parsing, no output parsing
    cmd = Command(2, myexp, input_parser=lambda x, y: (y, x))
    assert cmd(3, 10) == 1000

    # multi-input parsing *and* output parsing
    cmd = Command(2, myexp, input_parser=lambda x, y: (y, x),
                  output_parser=lambda x: 10 * x)
    assert cmd(8, 2) == 2560
