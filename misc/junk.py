from data_set import ParamSpec, DataSet,  param_spec
from qcodes.tests.instrument_mocks import MockParabola
import logging
logging.basicConfig(level="DEBUG")

NAME = "name"
HASH = "6ae999552a0d2dca14d62e2bc8b764d377b1dd6c"
SETPOINT_HASH = "7ae999552a0d2dca14d62e2bc8b764d377b1dd6c"
try:
    qc_param = param_spec(MockParabola("parabola"))
except:
    del qc_param
    pass
y = ParamSpec("y")
y.setpoints = qc_param.id
z = ParamSpec("z")
z.setpoints = y.id
values = [[1, 2, 3], [3, 4, 6], [7, 8, 9]]
d = DataSet(NAME, [qc_param, y, z], values)
# print(d.snapshot())


def callback(data, id, state):
    print(f"{data}@{id}")

sub_id = d.subscribe(callback, min_wait=1000)
import time
print('sleepin')
time.sleep(3)
print('done')
d.mark_complete()
import sys
sys.exit()
#d.unsubscribe(sub_id)
