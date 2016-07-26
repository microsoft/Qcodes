import time
import multiprocessing as mp


sleep_time = 0.001

_calibration = {
    'sleep_delay': None,
    'async_sleep_delay': None,
    'mp_start_delay': None
}


def calibrate(quiet=False):
    if _calibration['mp_start_delay'] is None:
        if not quiet:  # pragma: no cover
            print('multiprocessing startup delay and regular sleep delays:')
        mp_res = mptest(quiet=quiet)
        _calibration['blocking_time'] = abs(mp_res['blocking_time'])
        _calibration['mp_start_delay'] = abs(mp_res['startup_time'])
        _calibration['mp_finish_delay'] = abs(mp_res['finish_time'])
        _calibration['sleep_delay'] = abs(mp_res['median'])

    return _calibration


def report(startup_time, deviations,
           queue=None, quiet=False):  # pragma: no cover
    deviations.sort()
    mindev = deviations[0]
    avgdev = sum(deviations) / len(deviations)
    meddev = deviations[len(deviations) // 2]
    maxdev = deviations[-1]
    if not quiet:
        print('startup time: {:.3e}'.format(startup_time))
        print('min/med/avg/max dev: {:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(
            mindev, meddev, avgdev, maxdev))

    out = {
        'startup_time': startup_time,
        'min': mindev,
        'max': maxdev,
        'avg': avgdev,
        'median': meddev,
        'finish_time': time.time()
    }
    if queue:
        queue.put(out)
    return out


def sleeper(n, d, t0, timer, queue, quiet):  # pragma: no cover
    times = []
    startup_time = time.time() - t0
    for i in range(n):
        times.append(timer())
        time.sleep(d)

    deviations = [times[i] - times[i - 1] - d for i in range(1, len(times))]
    return report(startup_time, deviations, queue, quiet)


def mptest(n=100, d=0.001, timer=time.perf_counter, quiet=False):
    '''
    test time.sleep performance, and the time to start a multiprocessing
    Process. start time uses time.time() because some other timers start
    from zero in each new process

    n: how many asyncio.sleep calls to use
        default 100
    d: delay per sleep
        default 0.001
    timer: which system timer to use
        default time.perf_counter
    quiet: don't print anything
        default False
    '''

    q = mp.Queue()
    start_time = time.time()
    p = mp.Process(target=sleeper, args=(n, d, start_time, timer, q, quiet))
    p.start()
    blocking_time = time.time() - start_time
    p.join()

    out = q.get()
    out['finish_time'] = time.time() - out['finish_time']
    out['blocking_time'] = blocking_time
    return out
