from qcodes.config import Config

cfg = Config()
loc = cfg['GUID_components']['location']
station = cfg['GUID_components']['work_station']

if station == 0 or loc == 0:
    mssg = """
           Welcome to QCoDeS!
           This is the first-time-only configuration menu. When using the
           QCoDeS SQLite-based dataset (source files in qcodes/dataset/*), a
           Globally Unique IDentifier (GUID) is
           generated for each run that goes into the database. This is helpful
           when sharing data with other people or just yourself across
           different machines.
           To ensure the uniqueness, QCoDeS asks you to label each machine
           that runs QCoDeS using two integer codes:
             * one for your location
               (this could be 1 for your laboratory, 2 for the theory
                department, etc.)
             * one for your work station at that location
               (that could be 1 for the cryogenic control PC, 2 for
                the soldering station PC, etc.)
           These two codes will show up together with a timestamp
           for each run.
           """
    print(mssg)

if loc == 0:
    while loc < 1 or loc > 256:
        try:
            loc = int(input('Please enter an integer (1-256) as your'
                            ' location code: '))
            print(f'Location code is: {loc}')
        except ValueError:
            loc = 0
    cfg['GUID_components']['location'] = loc

if station == 0:
    while station < 1 or station > 16777216:
        try:
            station = int(input('Please enter an integer (1-16777216) as your'
                                ' work station code: '))
            print(f'Work station code is: {station}')
        except ValueError:
            station = 0
    cfg['GUID_components']['work_station'] = station

cfg.save_to_home()
