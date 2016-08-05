def setup_ESR_pulses(PulseBlaster,
                     loadpulse,
                     mwpulses,
                     mwtaos,
                     mwpost,
                     readpulse,
                     mwsource):

    #Constants
    PULSE_PROGRAM = 0

    CONTINUE = 0
    STOP = 1
    LOOP = 2
    END_LOOP = 3
    JSR = 4
    RTS = 5
    BRANCH = 6
    LONG_DELAY = 7
    WAIT = 8
    RTI = 9


    assert(len(mwpulses) == len(mwtaos)+1, 'Number of pulses in sequence does not correspond to number of wait times')

    # if nargin > 5 & & mwsource == 2
    #     pbout = 4;
    # else
    #     pbout = 2;
    # end

    #Convert input durations from ms to ns
    loadpulse *= 1e6
    mwpost *= 1e6
    mwpulses *= 1e6
    mwtaos *= 1e6
    readpulse *= 1e6

    PulseBlaster.select_board(0)
    PulseBlaster.start_programming(PULSE_PROGRAM)
    start = PulseBlaster.send_instruction(1, CONTINUE, 0, loadpulse-(sum(mwpulses)+sum(mwtaos)+mwpost))

    PulseBlaster.send_instruction(pbout, CONTINUE, 0, mwpulses(0))

    if len(mwtaos) > 0:
        for i, mwtao in enumerate(mwtaos):
            if mwtao >= 25:
                PulseBlaster.send_instruction(0, CONTINUE, 0, mwtaos)
            PulseBlaster.send_instruction(pbout, CONTINUE, 0, mwtaos[i+1])

    PulseBlaster.send_instruction(0, CONTINUE, 0, mwpost)
    PulseBlaster.send_instruction(1, CONTINUE, 0, readpulse/2)
    PulseBlaster.send_instruction(0, BRANCH, start, readpulse/2)
    PulseBlaster.send_instruction(1, CONTINUE, 0, readpulse/2)

    PulseBlaster.stop_programming()

    PulseBlaster.start()