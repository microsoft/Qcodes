import threading
import time


def sleeper(t, n, out):
    time.sleep(t)
    out[n] = n


def runmany(n, t):
    out = [None] * n

    t0 = time.perf_counter()

    threads = [threading.Thread(target=sleeper, args=(t, i, out))
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

    print('{} parallel threads sleeping'.format(n) +
          'given time: {}\n'.format(t) +
          'resulting time: {}\n'.format(t1 - t0) +
          '\n'.join(out_ok))
