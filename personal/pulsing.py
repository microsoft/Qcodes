def setup_ESR_pulses(PulseBlaster,
                     loadpulse,
                     mwpulses,
                     mwtaos,
                     mwpost,
                     readpulse,
                     mwsource=None):

    assert(len(mwpulses) == len(mwtaos)+1, 'Number of pulses in sequence does not correspond to number of wait times')

    pbout = 4 if mwsource == 2 else 2

    #Convert input durations from ms to ns
    loadpulse *= 1e6
    mwpost *= 1e6
    mwpulses *= 1e6
    mwtaos *= 1e6
    readpulse *= 1e6

    PulseBlaster.select_board(0)

    PulseBlaster.start_programming()
    start = PulseBlaster.send_instruction(1, 'continue', 0, loadpulse-(sum(mwpulses)+sum(mwtaos)+mwpost))

    PulseBlaster.send_instruction(pbout, 'continue', 0, mwpulses(0))

    if len(mwtaos) > 0:
        for i, mwtao in enumerate(mwtaos):
            if mwtao >= 25:
                PulseBlaster.send_instruction(0, 'continue', 0, mwtaos)
            PulseBlaster.send_instruction(pbout, 'continue', 0, mwtaos[i+1])

    PulseBlaster.send_instruction(0, 'continue', 0, mwpost)
    PulseBlaster.send_instruction(1, 'continue', 0, readpulse/2)
    PulseBlaster.send_instruction(0, 'branch', start, readpulse/2)

    PulseBlaster.stop_programming()

    PulseBlaster.start()