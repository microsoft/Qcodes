import multiprocessing as mp
import qcodes.tests.test_instrument as ti

mp.set_start_method('spawn')


class A:
    pass

a = A()

ti.TestParameters.setUp(a)
m = a.gates.connection.manager

print('active children', mp.active_children())
print('get chan2', a.gates.get('chan2'))
