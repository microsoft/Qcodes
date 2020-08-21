from qcodes.instrument.parameter import Parameter

def set_x(value):
    print('set x to %f' % value)

def test_setting_non_gettable_parameter():
    x = Parameter('x', initial_value=0, step=0.1, inter_delay=0.1, set_cmd=set_x)
    assert x.cache.get() == 0
    x.set(1)
    assert x.cache.get() == 1

