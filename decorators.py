import numpy as np

class printout(object):
    '''
    a class-based decorator - so it's picklable.
    '''
    def __init__(self, header):
        self.header = header

    def _out(self, *args, **kwargs):
        out = self.f(*args, **kwargs)
        print(self.header, out)
        return out

    def __call__(self, f):
        self.f = f
        return self._out


@printout('???')
def mult(a, b):
    return a * b


def add_to_numpy(key, val):
    np.edited_by_aj = True
    setattr(np, key, val)


if __name__ == '__main__':
    ret = mult(4, 4)
    print(ret)
