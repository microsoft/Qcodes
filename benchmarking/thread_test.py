import threading
import time


def sleeper(t, n, out, make_error):
    time.sleep(t)
    out[n] = n
    if make_error:
        raise RuntimeError('hello from # {}!'.format(n))


def runmany(n, t, error_nums):
    out = [None] * n

    t0 = time.perf_counter()

    threads = [
        CatchingThread(target=sleeper, args=(t, i, out, i in error_nums))
        for i in range(n)]

    # start threads backward
    [t.start() for t in reversed(threads)]
    [t.join() for t in threads]

    t1 = time.perf_counter()

    out_ok = []
    for i in range(n):
        if out[i] != i:
            out_ok += ['ERROR! out[{}] = {}'.format(i, out[i])]

    if not out_ok:
        out_ok += ['all output correct']

    print('{} parallel threads sleeping\n'.format(n) +
          'given time: {}\n'.format(t) +
          'resulting time: {}\n'.format(t1 - t0) +
          '\n'.join(out_ok))


class CatchingThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exception = e

    def join(self):
        super().join()
        if self.exception:
            raise self.exception
