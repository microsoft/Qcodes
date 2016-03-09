import time
import asyncio
import multiprocessing as mp


sleep_time = 0.001

_calibration = {
    'sleep_delay': None,
    'async_sleep_delay': None,
    'mp_start_delay': None,
    'timing_resolution': None
}


def calibrate(quiet=False):
    if _calibration['timing_resolution'] is None:
        if not quiet:
            print('Timing resolution:')
        _calibration['timing_resolution'] = abs(timertest(quiet=quiet))

    if _calibration['async_sleep_delay'] is None:
        if not quiet:
            print('async sleep delays:')
        _calibration['async_sleep_delay'] = abs(atest(quiet=quiet)['median'])

    if _calibration['mp_start_delay'] is None:
        if not quiet:
            print('multiprocessing startup delay and regular sleep delays:')
        mp_res = mptest(quiet=quiet)
        _calibration['blocking_time'] = abs(mp_res['blocking_time'])
        _calibration['mp_start_delay'] = abs(mp_res['startup_time'])
        _calibration['mp_finish_delay'] = abs(mp_res['finish_time'])
        _calibration['sleep_delay'] = abs(mp_res['median'])

    return _calibration


def report(startup_time, deviations, queue=None, quiet=False):
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
    # call sleeper once locally, just for
    # sleeper(1, 0, 0, timer, mp.Queue(), True)

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


@asyncio.coroutine
def asleeper(n, d, t0, timer, queue, quiet):
    times = []
    startup_time = time.time() - t0
    for i in range(n):
        times.append(timer())
        yield from asyncio.sleep(d)

    deviations = [times[i] - times[i - 1] - d for i in range(1, len(times))]
    return report(startup_time, deviations, queue)


def atest(n=100, d=sleep_time, timer=time.perf_counter, quiet=False):
    '''
    test asyncio.sleep performance, and the time to start
    an async event loop. start time uses time.time() for
    consistency with mptest, rather than the timer argument

    n: how many asyncio.sleep calls to use
        default 100
    d: delay per sleep
        default 0.001 (timing.sleep_time, so other routines can share it)
    timer: which system timer to use
        default time.perf_counter
    quiet: don't print anything
        default False
    '''
    q = mp.Queue()
    t0 = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asleeper(n, d, t0, timer, q, quiet))
    return q.get()


def timertest(total_time=1, timer=time.perf_counter, quiet=False):
    '''
    test the precision of a timer, by looking at the minimum nonzero time
    difference it reports between return values

    total_time: how long to aggregate for
        default 1 (second)
    timer: which system timer to use
        default time.perf_counter
    quiet: don't print anything
        default False
    '''
    now = timer()
    start = now
    deviations = []
    while now - start < total_time:
        last = now
        while now == last:
            now = timer()
        deviations.append(now - last)

    out = report(0, deviations, quiet=quiet)
    if(out['median'] / out['min'] > 2):  # pragma: no cover
        raise RuntimeError('median is too high vs. minimum')

    return out['min']
