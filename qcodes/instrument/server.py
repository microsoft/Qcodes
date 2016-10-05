import multiprocessing as mp

from qcodes.process.server import ServerManager, BaseServer


def get_instrument_server_manager(server_name, shared_kwargs={}):
    """
    Find or make a given `InstrumentServerManager`.

    An `InstrumentServer` holds one or more Instrument objects, and an
    `InstrumentServerManager` allows other processes to communicate with this
    `InstrumentServer`.

    Both the name and the shared attributes must match exactly. If no manager
    exists with this name, it will be created with the given `shared_kwargs`.
    If an manager exists with this name but different `shared_kwargs` we
    raise an error.

    server_name: (default 'Instruments') which server to put the instrument on.
        If a server with this name exists, the instrument will be added to it.
        If not, a new server is created with this name.

    shared_kwargs: unpicklable items needed by the instruments on the
        server, will get sent with the manager when it's started up
        and included in the kwargs to construct each new instrument
    """
    if not server_name:
        server_name = 'Instruments'

    instances = InstrumentServerManager.instances
    manager = instances.get(server_name, None)

    if manager and manager._server in mp.active_children():
        if shared_kwargs and manager.shared_kwargs != shared_kwargs:
            # it's OK to add another instrument that has  *no* shared_kwargs
            # but if there are some and they're different from what's
            # already associated with this server, that's an error.
            raise ValueError(('An InstrumentServer with name "{}" already '
                              'exists but with different shared_attrs'
                              ).format(server_name))
    else:
        manager = InstrumentServerManager(server_name, shared_kwargs)

    return manager


class InstrumentServerManager(ServerManager):
    """
    Creates and manages connections to an InstrumentServer

    name: the name of the server to create
    kwargs: extra items to send to the server on creation (such as
        additional queues, that can only be shared on creation)
        These items will be set as attributes of any instrument that
        connects to the server
    """
    instances = {}

    def __init__(self, name, shared_kwargs=None):
        self.name = name
        self.shared_kwargs = shared_kwargs
        self.instances[name] = self

        self.instruments = {}

        super().__init__(name=name, server_class=InstrumentServer,
                         shared_attrs=shared_kwargs)

    def restart(self):
        """
        Restart the InstrumentServer and reconnect the instruments that
        had been connected previously
        """
        super().restart()

        instruments = self.instruments.values()
        self.instruments = {}
        for instrument in instruments:
            instrument.connect()

    def connect(self, remote_instrument, instrument_class, args, kwargs):
        new_id = self.ask('new_id')
        try:
            info = self.ask('new', instrument_class, new_id, *args, **kwargs)
            self.instruments[new_id] = remote_instrument

        except:
            # if anything went wrong adding a new instrument, delete it
            # in case it still exists there half-formed.
            self.delete(new_id)
            raise

        return info

    def delete(self, instrument_id):
        self.write('delete', instrument_id)

        if self.instruments.get(instrument_id, None):
            del self.instruments[instrument_id]

            if not self.instruments:
                self.close()
                self.instances.pop(self.name, None)


class InstrumentServer(BaseServer):
    # just for testing - how long to allow it to wait on a queue.get
    timeout = None

    def __init__(self, query_queue, response_queue, shared_kwargs):
        super().__init__(query_queue, response_queue, shared_kwargs)

        self.instruments = {}
        self.next_id = 0

        # Ensure no references of instruments defined in the main process
        # are copied to the server process. With the spawn multiprocessing
        # method this is not an issue, as the class is reimported in the
        # new process, but with fork it can be a problem ironically.
        from qcodes.instrument.base import Instrument
        Instrument._all_instruments = {}

        self.run_event_loop()

    def handle_new_id(self):
        """
        split out id generation from adding an instrument
        so that we can delete it if something goes wrong!
        """
        new_id = self.next_id
        self.next_id += 1
        return new_id

    def handle_new(self, instrument_class, new_id, *args, **kwargs):
        """
        Add a new instrument to the server.

        After the initial load, the instrument is referred to by its ID.

        Args:
            instrument_class (class): The type of instrument to construct.

            new_id (int): The ID by which this instrument will be known on the
                server.

            *args: positional arguments to the instrument constructor.

            **kwargs: keyword arguments to the instrument constructor.

        Returns:
            dict: info to reconstruct this instrument's API in the remote.
                See ``Instrument.connection_attrs`` for details.
        """

        # merge shared_kwargs into kwargs for the constructor,
        # but only if this instrument_class is expecting them.
        # The *first* instrument put on a given server must have
        # all the shared_kwargs sent with it, but others may skip
        # (for now others must have *none* but later maybe they could
        # just skip some of them)
        for key, value in self._shared_attrs.items():
            if key in instrument_class.shared_kwargs:
                kwargs[key] = value
        ins = instrument_class(*args, server_name=None, **kwargs)

        self.instruments[new_id] = ins

        # info to reconstruct the instrument API in the RemoteInstrument
        return ins.connection_attrs(new_id)

    def handle_delete(self, instrument_id):
        """
        Delete an instrument from the server, and stop the server if their
        are no more instruments left after this.
        """
        if instrument_id in self.instruments:
            self.instruments[instrument_id].close()

            del self.instruments[instrument_id]

            if not any(self.instruments):
                self.handle_halt()

    def handle_cmd(self, instrument_id, func_name, *args, **kwargs):
        """
        Run some method of an instrument
        """
        func = getattr(self.instruments[instrument_id], func_name)
        return func(*args, **kwargs)
