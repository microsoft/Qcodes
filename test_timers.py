import time
for clock in ['clock', 'monotonic', 'perf_counter', 'process_time', 'time']:
    print(f'{clock}: {time.get_clock_info(clock)}')
