from data_set import ParamSpec, DataSet,  param_spec
from qcodes.tests.instrument_mocks import MockParabola, MultiSetPointParam
import logging
import time


logging.basicConfig(level="DEBUG")

NAME = "name"
p = MockParabola("parabola")
qc_param = param_spec(p)
x = ParamSpec("x")
qc_param.setpoints = x.id
d = DataSet(NAME, [x, qc_param])
d.add_metadata("time_started", time.time())


class memory_serializer():

    def __init__(self, d):
        self.parameters = d.get_parameters()
        self.name = d.name
        self.d = d
        self.data = []
        # data index
        self.last_index = 0

    def callback(self, data, len, state):
        rows = data.get_data(*self.parameters, start=self.last_index)
        self.last_index = len
        for row in zip(*rows):
            self.data.append(row)
        # state.get(somethign)
        # do somethign else maybe?


# # 1d loop
in_mem = memory_serializer(d)
sub_id = d.subscribe(in_mem.callback, state=d.metadata)
for i in range(1001):
    # set a value
    d.add_result("x", i)
    # get a value
    d.add_result(p.parabola.name, p.parabola())
d.add_metadata("time_ended", time.time())
d.mark_complete()
print(d)
print(d.snapshot())


# 2d  soft loop
NAME = "xyz"
x = ParamSpec("x")
y = ParamSpec("y")
y.setpoints = x.id
z = ParamSpec("z")
z.setpoints = y.id

d = DataSet(NAME, [x, y, z])
d.add_metadata("time_started", 10)
in_mem = memory_serializer(d)
sub_id = d.subscribe(in_mem.callback, state=d.metadata)
for i in range(10):
    # set a value
    d.add_result("x", i)
    for j in range(10):
        d.add_result("y", j)
        # get a value
        d.add_result('z', j+i)
        time.sleep(0.1)

d.add_metadata("time_ended", time.time())
d.mark_complete()
print(d)
print(d.to_array("z", (10, 10)))
print(d.snapshot())

# 2d  hard loop
# not clear from the spec how one WANTS to save this