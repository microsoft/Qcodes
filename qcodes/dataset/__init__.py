from qcodes.config import Config

cfg = Config()
loc = cfg['GUID_components']['location']
station = cfg['GUID_components']['work_station']

if station == 0 or loc == 0:
    mssg = """
           Welcome to QCoDeS!
           This is the first-time-only configuration menu. When using the
           QCoDeS dataset (source files in qcodes/dataset/*), a GUID is
           generated for each run that goes into the database. This is helpful
           when sharing data with other people or just yourself across
           different machines. To enable the GUID generation, please provide
           one or two input integers to identify your machine.
           """
    print(mssg)

if loc == 0:
    while loc < 1 or loc > 256:
        loc = int(input('Please enter an integer (1-256) as your'
                        ' location code: '))
    cfg['GUID_components']['location'] = loc

if station == 0:
    while station < 1 or station > 16777216:
        station = int(input('Please enter an integer (1-16777216) as your work'
                            ' station code: '))
    cfg['GUID_components']['work_station'] = station

cfg.save_to_home()