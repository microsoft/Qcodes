" MWE of meta instrument (with debugging enabled)  "
import concurrent.futures as futures
import logging
import multiprocessing as mp
import time

from functools import partial

from qcodes import Instrument


# agreesive logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s- %(message)s')

class MyInstrument(Instrument):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.value=0
        self.add_parameter('x',  get_cmd=self.getx, set_cmd=self.setx)

    def getx(self):
        return self.value

    def setx(self, val):
        logging.debug("set {}".format(val))
        self.value = val
        # simulate delay 5 seconds
        time.sleep(5)
        logging.debug("done {}".format(val))
        return


class Meta(Instrument):
    shared_kwargs = ['instruments']

    # Instruments will be a list of RemoteInstrument objects, which can be
    # given to a server on creation but not later on, so it needs to be
    # listed in shared_kwargs

    def __init__(self, name, instruments=(), **kwargs):
        super().__init__(name, **kwargs)
        self._instrument_list = instruments
        self.no_instruments = len(instruments)
        for gate in range(len(self._instrument_list)):
            self.add_parameter('c%d' % gate,
                               get_cmd=partial(self._get, gate=gate),
                               set_cmd=partial(self._set, gate=gate))

        self.add_parameter("setBoth", set_cmd=partial(self._set_both))
        self.add_parameter("setBothAsync", set_cmd=partial(self._set_async))

    def _set_both(self, value):
        for i in self._instrument_list:
            i.set('x', value)

    def _set_async(self, value):
        with futures.ThreadPoolExecutor(max_workers=self.no_instruments+10) as executor:
            jobs = []
            for i in self._instrument_list:
                job = executor.submit(partial(i.set, 'x'), value)
                jobs.append(job)

            futures.wait(jobs)

    def _get(self, gate):
        value =self._instrument_list[gate].get('x')
        logging.debug('Meta get gate %s' % (value))
        return value

    def _set(self, value, gate):
        logging.debug('Meta set gate %s @ value %s' % (gate, value))
        i = self._instrument_list[gate]
        i.set('x', value)



if __name__ == '__main__':

    mp.set_start_method('spawn')

    base1 = MyInstrument(name='zero', server_name="foo")
    base2 = MyInstrument(name='one', server_name="bar")

    meta_server_name = "meta_server"
    meta = Meta(name='meta', server_name=meta_server_name,
                      instruments=[base1, base2])


    print("--- set meta --- ")
    meta.c1.set(25)
    print(meta.c1.get())
    print(base1.x.get())

    print("--- set base --- ")
    base1.x.set(1)
    print(meta.c1.get())
    print(base1.x.get())


    print("--- both --- ")
    meta.setBoth(0)
    print(base2.x.get())
    print(base1.x.get())

    print("--- no block --- ")
    meta.setBothAsync(10)
    logging.debug(base1.x.get())
