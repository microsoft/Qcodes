def setup_ESR_pulses(PulseBlaster,
                     load_pulse,
                     mw_pulses,
                     mw_delays,
                     mw_final_delay,
                     readpulse_duration,
                     mw_source=None):

    assert(len(mw_pulses) == len(mw_delays) + 1,
           'Number of pulses in sequence does not correspond to number of wait times')

    output_port = 4 if mw_source == 2 else 2

    # Convert input durations from ms to ns
    load_pulse *= 1e6
    mw_final_delay *= 1e6
    mw_pulses *= 1e6
    mw_delays *= 1e6
    readpulse_duration *= 1e6

    PulseBlaster.detect_boards()
    PulseBlaster.select_board(0)

    # Start programming the PulseBlaster
    PulseBlaster.start_programming()
    cycles = load_pulse - (sum(mw_pulses) + sum(mw_delays) + mw_final_delay)
    start = PulseBlaster.send_instruction(1, 'continue', 0, cycles)

    PulseBlaster.send_instruction(output_port, 'continue', 0, mw_pulses(0))

    if len(mw_delays) > 0:
        for i, mw_delay in enumerate(mw_delays):
            if mw_delay >= 25:
                PulseBlaster.send_instruction(0, 'continue', 0, mw_delays)
            PulseBlaster.send_instruction(output_port, 'continue', 0, mw_delays[i + 1])

    PulseBlaster.send_instruction(0, 'continue', 0, mw_final_delay)
    PulseBlaster.send_instruction(1, 'continue', 0, readpulse_duration / 2)
    PulseBlaster.send_instruction(0, 'branch', start, readpulse_duration / 2)

    PulseBlaster.stop_programming()

    # Send a software trigger to PulseBlaster
    PulseBlaster.start()
