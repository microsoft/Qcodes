# stress test multiprocessing
# run with: python mptest.py <proc_count> <period> <repetitions> <start_method>
# proc_count: the number of processes to spin up and load qcodes
# period: milliseconds to wait between calling each process
# repetitions: how many times to call each one
# start_method: multiprocessing method to use (fork, forkserver, spawn)

import multiprocessing as mp
import sys
import os
import psutil
import time

import qcodes as qc


timer = time.perf_counter


def mp_test(name, qin, qout, qglobal):
    '''
    simple test that keeps a process running until asked to stop,
    and looks for an attribute within qcodes just to ensure it has loaded it
    '''
    delays = []
    first = True
    while True:
        item, qtime = qin.get()
        if first:  # ignore the first one... process is still starting
            first = False
        else:
            delays.append(timer() - qtime)
        if item == 'break':
            qglobal.put({
                'name': name,
                'avg': sum(delays) / len(delays),
                'max': max(delays)
            })
            break
        qout.put(repr(getattr(qc, item)))


def get_memory(pid):
    mi = psutil.Process(pid).memory_info()
    return mi.rss, mi.vms


def get_all_memory(processes, title):
    main_memory = get_memory(os.getpid())
    proc_memory = [0, 0]
    for proc in processes:
        for i, v in enumerate(get_memory(proc.pid)):
            proc_memory[i] += v
            # print(v)

    return {
        'main_physical': main_memory[0]/1e6,
        'main_virtual': main_memory[1]/1e6,
        'proc_physical': proc_memory[0]/1e6,
        'proc_virtual': proc_memory[1]/1e6,
        'title': title
    }


def format_memory(mem):
    return ('{title}\n'
            '  main:  {main_physical:.0f} MB phys, '
            '{main_virtual:.0f} MB virt\n'
            '  procs: {proc_physical:.0f} MB phys, '
            '{proc_virtual:.0f} MB virt\n'
            '').format(**mem)


if __name__ == '__main__':
    proc_count = int(sys.argv[-4])
    period = float(sys.argv[-3])
    reps = int(sys.argv[-2])
    method = sys.argv[-1]
    mp.set_start_method(method)

    qglobal = mp.Queue()

    mem = [get_all_memory([], 'on startup')]

    queues = []
    processes = []
    resp_delays = []

    t_before_start = timer()

    for proc_num in range(proc_count):
        qin = mp.Queue()
        qout = mp.Queue()
        queues.append((qin, qout))
        p = mp.Process(target=mp_test,
                       args=('p{}'.format(proc_num), qin, qout, qglobal))
        processes.append(p)
        p.start()

    start_delay = (timer() - t_before_start) * 1000

    mem.append(get_all_memory(processes, 'procs started'))

    for i in range(reps):
        for qin, qout in queues:
            t1 = timer()
            qin.put(('Loop', timer()))
            loop_repr = qout.get()
            if i:
                # ignore the first one, process is still starting
                resp_delays.append((timer() - t1) * 1000)
            if(loop_repr != repr(qc.Loop)):
                raise RuntimeError('{} != {}'.format(loop_repr, repr(qc.Loop)))
        print('.', end='', flush=True)
        time.sleep(period / 1000)
    print('')

    mem.append(get_all_memory(processes, 'procs done working'))

    for qin, qout in queues:
        qin.put(('break', timer()))

    t_before_join = timer()
    for proc in processes:
        proc.join()
    join_delay = (timer() - t_before_join) * 1000

    delays = [qglobal.get() for proc in processes]
    avg_delay = sum([d['avg'] for d in delays]) * 1000 / len(delays)
    max_delay = max([d['max'] for d in delays]) * 1000

    avg_resp_delay = sum(resp_delays) / len(resp_delays)
    max_resp_delay = max(resp_delays)

    print(('Ran {} procs using "{}" method\n'
           'sent messages every {} milliseconds, {} times\n'
           ).format(proc_count, method, period, reps))

    print('Milliseconds to start all processes: {:.3f}'.format(start_delay))
    print('Final join delay: {:.3f}\n'.format(join_delay))

    print('Milliseconds to receive to queue request')
    print('  avg: {:.6f}'.format(avg_delay))
    print('  max: {:.6f}\n'.format(max_delay))

    print('Milliseconds to respond to queue request')
    print('  avg: {:.6f}'.format(avg_resp_delay))
    print('  max: {:.6f}\n'.format(max_resp_delay))

    # report on the memory results
    for m in mem:
        print(format_memory(m))
