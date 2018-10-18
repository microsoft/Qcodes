from qcodes import VisaInstrument


class KeysightB220X(VisaInstrument):
    """
    QCodes driver for B2200 / B2201 switch matrix
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r', **kwargs)

    """
    [:ROUT]:FUNC channel_config
    [:ROUT]:FUNC?
    
    Sets the channel configuration mode.
    channel_config:
    ACON: Auto Config Mode
    NCON: Normal Config Mode
    Query returns the present channel configuration: ACON or NCON.
    """


    """
    [:ROUT]:CONN:RULE card_number,rule
    [:ROUT]:CONN:RULE? card_number
    
    Sets the connection rule for the specified card. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    rule: FREE (free) or SROUte (single)
    Query returns the connection rule of the specified card: FREE or SROU.
    card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """


    """
    [:ROUT]:CONN:SEQ card_number,sequence
    [:ROUT]:CONN:SEQ? card_number
    
    
    Specifies the connection sequence mode for the specified card.
    card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config sequence:
    NSEQ: No-Sequence mode
    BBM: Break-Before-Make mode MBBR: Make-Before-Break mode
    Query returns the connections sequence mode of the specified card: NSEQ, BBM, or MBBR. card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """


    """
    [:ROUT]:SYMB:CHAN card_number,channel,'string'
    [:ROUT]:SYMB:CHAN? card_number,channel
    
    Defines a string for the specified channel. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    channel: channel number, 1 to 48 for Auto, 1 to
    12 for Normal
    Query returns the symbol string set to the specified channel.
    """


    """
    [:ROUT]:SYMB:PORT port,'string'
    [:ROUT]:SYMB:PORT? port
    
    Defines a string for the specified input port. port: input port number, 1 to 14
    Query returns the symbol string set to the specified input port.
    """

    # Relay Control Commands

    """
    [:ROUT]:OPEN:CARD card_number
    Disconnects all input ports from all output ports for the specified card.
    card_number: 0 or ALL for Auto Config, 1, 2, 3, 4, or ALL for Normal Config
    """


    """
    [:ROUT]:OPEN[:LIST] (@channel_list)
    [:ROUT]:OPEN[:LIST]? (@channel_list)
    
    Disconnects the input ports from output ports as specified in channel_list.
    channel_list: Channels to open.
    Query returns the status of the specified channels: 0 (closed) or 1 (opened).
    channel_list: Channels to check.
    """


    """
    [:ROUT]:CLOS:CARD? card_number
    Returns channel_list of all closed channels for the specified card. “closed channel” means an input port connected to an output port.
    card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """


    """
    [:ROUT]:CLOS[:LIST] (@channel_list)
    [:ROUT]:CLOS[:LIST]? (@channel_list)
    
    Connects the input ports to the output ports as specified in channel_list.
    channel_list: Channels to close.
    Query returns the status of the specified channels: 1 (closed) or 0 (opened).
    channel_list: Channels to check.
    """

    # Bias Mode Commands
    """
    [:ROUT]:BIAS:CHAN:DIS:CARD card_number
    Bias-disables the specified card. card_number: 0 or ALL for Auto Config, 1, 2, 3, 4, or ALL for Normal Config
    """

    """
    [:ROUT]:BIAS:CHAN:DIS[:LIST] (@channel_list)
    [:ROUT]:BIAS:CHAN:DIS[:LIST]? (@channel_list)
    Bias-disables the specified channels. channel_list: Channels to bias-disable.
    Query returns the status of the specified channels: 1 (disabled) or 0 (enabled).
    channel_list: Channels to check.
    """


    """
    [:ROUT]:BIAS:CHAN:ENAB:CARD card_number
    Bias-enables the specified card. card_number: 0 or ALL for Auto Config, 1, 2, 3, 4, or ALL for Normal Config
    """


    """
    [:ROUT]:BIAS:CHAN:ENAB[:LIST] (@channel_list)
    [:ROUT]:BIAS:CHAN:ENAB[:LIST]? (@channel_list)
    Bias-enables the specified channels. channel_list: Channels to bias-enable.
    Query returns the status of the specified channels: 1 (enabled) or 0 (disabled).
    channel_list: Channels to check.
    """


    """
    [:ROUT]:BIAS:PORT card_number,bias_port
    [:ROUT]:BIAS:PORT? card_number
    Specifies the input Bias Port for the specified card. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    bias_port: 1 to 14 or -1
    Query returns the input Bias Port number for the specified card.
    card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """


    """
    [:ROUT]:BIAS[:STAT] card_number,state
    [:ROUT]:BIAS[:STAT]? card_number
    
    Sets the bias mode for the specified card. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    state: ON / 1 (mode ON) or OFF / 0 (mode OFF)
    Query returns the mode status of the specified card: 0 (OFF) or 1 (ON).
    card_number: Card to check. 0 for Auto Config,
    1, 2, 3, or 4 for Normal Config.
    """

    # Ground Mode Commands
    """
    [:ROUT]:AGND:CHAN:DIS:CARD card_number
    Ground-disables the specified card. card_number: 0 or ALL for Auto Config, 1, 2, 3, 4, or ALL for Normal Config
    """

    """
    [:ROUT]:AGND:CHAN:DIS[:LIST] (@channel_list)
    [:ROUT]:AGND:CHAN:DIS[:LIST]? (@channel_list)
    Ground-disables the specified channels. channel_list: Channels to ground-disable.
    Query returns the status of the specified channels: 1 (disabled) or 0 (enabled).
    channel_list: Channels to check.
    """


    """
    [:ROUT]:AGND:CHAN:ENAB:CARD card_number
    Ground-enables the specified card. card_number: 0 or ALL for Auto Config, 1, 2, 3, 4, or ALL for Normal Config
    """


    """
    [:ROUT]:AGND:CHAN:ENAB[:LIST] (@channel_list)
    [:ROUT]:AGND:CHAN:ENAB[:LIST]? (@channel_list)
    Ground-enables the specified channels. channel_list: Channels to ground-enable.
    Query returns the status of the specified channels: 1 (enabled) or 0 (disabled).
    channel_list: Channels to check.
    """


    """
    [:ROUT]:AGND:PORT card_number,ground_port
    [:ROUT]:AGND:PORT? card_number
    Specifies the input Ground Port for the specified card.
    card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config ground_port: 1 to 14 or -1
    Query returns the input Ground Port number for the specified card.
    card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """


    """
    [:ROUT]:AGND[:STAT] card_number,state
    [:ROUT]:AGND[:STAT]? card_number
    Sets the ground mode for the specified card. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    state: ON / 1 (mode ON) or OFF / 0 (mode OFF)
    Query returns the mode status of the specified card: 0 (OFF) or 1 (ON).
    card_number: Card to check. 0 for Auto Config,
    1, 2, 3, or 4 for Normal Config.
    """


    """
    [:ROUT]:AGND:UNUSED card_number,'enable_port'
    [:ROUT]:AGND:UNUSED? card_number
    Ground-enables the specified input ports for the specified card.
    card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config enable_port: One or more input port numbers: 1 to 8. Enclose by single quotation marks. Separate multiple input port numbers by comma. For example: '1,5'
    Query returns the ground-enabled input port numbers for the specified card.
    card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """

    # Couple Mode Commands
    """
    [:ROUT]:COUP:PORT card_number,'couple_port'
    [:ROUT]:COUP:PORT? card_number
    Specifies the input couple ports for the specified card.
    card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    couple_port: One or more input port numbers: 1, 3, 5, 7, 9, 11, or 13. Enclose by single quotation marks. Separate multiple input port numbers by comma. For example: '1,5'
    Query returns the lower input port number of each couple pair for the specified card.
    card_number: Card to check. 0 for Auto Config, 1, 2, 3, or 4 for Normal Config.
    """

    """
    [:ROUT]:COUP:PORT:DET
    Detects the input ports connected to Kelvin cable, and assigns them as the input couple ports for the all cards.
    """

    """
    [:ROUT]:COUP[:STAT] card_number,state
    [:ROUT]:COUP[:STAT]? card_number
    Sets the couple mode for the specified card. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    state: ON / 1 (mode ON) or OFF / 0 (mode OFF)
    Query returns the mode status of the specified card: 0 (OFF) or 1 (ON).
    card_number: Card to check. 0 for Auto Config,
    1, 2, 3, or 4 for Normal Config.
    """

    # DIAG subsystem
    """
    :DIAG:TEST:CARD:CLE card_number
    Clears relay test result (pass/fail) of the specified card. card_number: 1, 2, 3, 4, or ALL
    """


    """
    :DIAG:TEST:CARD[:EXEC]? card_number
    Executes relay test, then returns result: 1 (fail card exists), 0 (pass).
    card_number: 1, 2, 3, 4, or ALL
    """


    """
    :DIAG:TEST:CARD:STAT? card_number
    Returns most recent relay test result: 1 (fail), 0 (pass), -1 (not tested).
    card_number: 1, 2, 3, 4
    """

    """
    :DIAG:TEST:FRAM:CLE item
    Clears specified test result.
    item: CONT (controller test), FPAN (front panel interface test), LED, PEN, or BEEP
    """


    """
    :DIAG:TEST:FRAM[:EXEC]? item
    Executes specified test, then returns test result: 1 (fail), 0 (pass).
    item: CONT (controller test), FPAN (front panel interface test), LED, PEN, or BEEP
    """


    """
    :DIAG:TEST:FRAM:STAT? item
    Returns most recent test result of the specified test: 1 (fail), 0 (pass), -1 (not tested).
    item: CONT (controller test), FPAN (front panel interface test), LED, PEN, or BEEP
    """

    # SYSTEM Subsystem

    """
    :SYST:BEEP state
    Enables/disables the beeper.
    state: ON / 1 (enable) or OFF / 0 (disable)
    """


    """
    :SYST:CCON? card_number
    Returns the card configuration information. This command is just to keep compatibility with the Keysight E5250A. card_number: 1, 2, 3, or 4
    """


    """
    :SYST:CDES? card_number
    Returns a description of the specified card: model number and input/output port information.
    card_number: Card to check. 0 for Auto Config,
    1, 2, 3, or 4 for Normal Config.
    """


    """
    :SYST:CPON card_number
    Resets the specified card to the power-on state. card_number: 0 or ALL for Auto Config,
    1, 2, 3, 4, or ALL for Normal Config
    """


    """
    :SYST:CTYP? card_number
    Returns ID of the specified card: model number and revision.
    card_number: Card to check. 0 for Auto Config,
    1, 2, 3, or 4 for Normal Config.
    """


    """
    :SYST:DISP:LCD state
    Enables/disables the front panel LCD when the B2200 is in the GPIB remote mode. state: ON / 1 (enable) or OFF / 0 (disable)
    """


    """
    :SYST:DISP:LED state
    Enables/disables the front panel LED. state: ON / 1 (enable) or OFF / 0 (disable)
    """

    """
    :SYST:DISP:STR string
    Specifies a string displayed on the LCD in the GPIB remote mode.
    """


    """
    :SYST:ERR?
    Reads error from head of error queue, and removes it from the queue.
    """


    """
    :SYST:KLC state
    Locks/unlocks the front panel keys. state: ON / 1 (lock) or OFF / 0 (unlock)
    """


    """
    :SYST:MEMO:SAVE memory_number :SYST:MEMO:LOAD memory_number
    Saves a setup information into the internal memory, or loads a setup information.
    memory_number: 1 to 8
    """


    """
    :SYST:MEMO:COMM memory_number,'comment' :SYST:MEMO:COMM? memory_number
    Memorizes the comment for the B2200 setup information specified by memory_number.
    memory_number: 1 to 8
    """


    """
    :SYST:MEMO:DEL memory_number
    Deletes the B2200 setup information and the comment specified by memory_number.
    memory_number: 1 to 8
    """


    """
    :SYST:PEN state
    Enables/disables the light pen.
    state: ON / 1 (enable) or OFF / 0 (disable)
    """


    """
    :SYST:VERS?
    Returns SCPI version number for which the B2200 complies.
    """
