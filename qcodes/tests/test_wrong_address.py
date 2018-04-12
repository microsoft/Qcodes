import logging

import qcodes.instrument.sims as sims
from qcodes.instrument_drivers.Keysight.Keysight_34465A import Keysight_34465A
visalib = sims.__file__.replace('__init__.py', 'Keysight_34465A.yaml@sim')

logging.basicConfig(level=logging.DEBUG)


def test_wrong_address():
    try:
        logging.warning('1')
        logging.warning(Keysight_34465A.instances())
        try:
            # wrong address
            keysight_sim = Keysight_34465A('keysight_34465A_sim',
                                           address='GPIB::1::6553512341234::INSTR',
                                           visalib=visalib)
        except Exception as e:
            logging.warning('this should be raised')
            logging.warning(e)
            logging.warning('2')
            logging.warning(Keysight_34465A.instances())
            pass
        # right address
        keysight_sim = Keysight_34465A('keysight_34465A_sim',
                                       address='GPIB::1::65535::INSTR',
                                       visalib=visalib)
        logging.warning('3')
        logging.warning(Keysight_34465A.instances())

    except Exception as e:
        logging.warning(e)
        logging.warning('not good')
        raise e
