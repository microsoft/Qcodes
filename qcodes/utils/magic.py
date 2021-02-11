from IPython.core.magic import Magics, magics_class, line_cell_magic

@magics_class
class QCoDeSMagic(Magics):
    """Magics related to code management (loading, saving, editing, ...)."""

    def __init__(self, *args, **kwargs):
        """
        Setup Magic. All args and kwargs are passed to super class.
        """
        self._knowntemps = set()
        super().__init__(*args, **kwargs)

    @line_cell_magic
    def measurement(self, line, cell=None):
        """
        Create ``qcodes.Loop`` measurement mimicking Python ``for`` syntax via
        iPython magic.

        Upon execution of a notebook cell, the code is transformed from the
        for loop structure to a QCoDeS Loop before being executed.
        Can be run by having ``%%measurement`` in the first line of a cell,
        followed by the measurement name (see below for an example).

        The for loop syntax differs slightly from a Python ``for`` loop,
        as it uses ``for {iterable}`` instead of ``for {element} in {iterable}``.
        The reason is that ``{element}`` cannot be accessed (yet) in QCoDeS loops.

        Comments (#) are ignored in the loop.
        Any code after the loop will also be run, if separated by a blank
        line from the loop.

        The Loop object is by default stored in a variable named ``loop``,
        and the dataset in ``data``, and these can be overridden using options.
        Must be run in a Jupyter Notebook.
        Delays can be provided in a loop by adding ``-d {delay}`` after ``for``.

        The following options can be passed along with the measurement name
        (e.g. ``%%measurement -px -d data_name {measurement_name})``::

            -p : print transformed code
            -x : Do not execute code
            -d <dataset_name> : Use custom name for dataset
            -l <loop_name> : Use custom name for Loop

        An example for a loop cell is as follows::

            %%measurement {-options} {measurement_name}
            for {sweep_vals}:
                {measure_parameter1}
                {measure_parameter2}
                for -d 1 {sweep_vals2}:
                    {measure_parameter3}

            ...

        which will be internally transformed to::

            import qcodes
            loop = qcodes.Loop({sweep_vals}).each(
                {measure_parameter1},
                {measure_parameter2},
                qcodes.Loop({sweep_vals2}, delay=1).each(
                    {measure_parameter3}))
            data = loop.get_data_set(name={measurement_name})

            ...

        An explicit example of the line ``for {sweep_vals}:`` could be
        ``for sweep_parameter.sweep(0, 42, step=1):``

        """

        if cell is None:
            # No loop provided, print documentation
            print(self.measurement.__doc__)
            return

        # Parse line, get measurement name and any possible options
        options, msmt_name = self.parse_options(line, 'pd:l:x')
        data_name = options.get('d', 'data')
        loop_name = options.get('l', 'loop')

        lines = cell.splitlines()
        assert lines[0][:3] == 'for', "Measurement must start with for loop"

        contents = f'import qcodes\n{loop_name} = '
        previous_level = 0
        for k, line in enumerate(lines):
            line, level = line.lstrip(), int((len(line)-len(line.lstrip())) / 4)

            if not line:
                # Empty line, end of loop
                break
            elif line[0] == '#':
                # Ignore comment
                continue
            else:
                line_representation = ' ' * level * 4
                if level < previous_level:
                    # Exiting inner loop, close bracket
                    line_representation += '),' * (previous_level - level)
                    line_representation += '\n' + ' ' * level * 4

                if line[:3] == 'for':
                    # New loop
                    for_opts, for_code = self.parse_options(line[4:-1], 'd:')
                    if 'd' in for_opts:
                        # Delay option provided
                        line_representation += ('qcodes.Loop({}, '
                                                'delay={}).each(\n'
                                                ''.format(for_code,
                                                          for_opts["d"]))
                    else:
                        line_representation += ('qcodes.Loop({}).each(\n'
                                                ''.format(for_code))
                else:
                    # Action in current loop
                    line_representation += f'{line},\n'
                contents += line_representation

                # Remember level for next iteration (might exit inner loop)
                previous_level = level

        # Add closing brackets for any remaining loops
        contents += ')' * previous_level + '\n'
        # Add dataset
        contents += "{} = {}.get_data_set(name='{}')".format(data_name,
                                                             loop_name,
                                                             msmt_name)

        for line in lines[k+1:]:
            contents += '\n' + line

        if 'p' in options:
            print(contents)

        if 'x' not in options:
            # Execute contents
            self.shell.run_cell(contents, store_history=True, silent=True)


def register_magic_class(cls=QCoDeSMagic, magic_commands=True):
    """
    Registers a iPython magic class.

    Args:
        cls: Magic class to register.
        magic_commands (List): List of magic commands within the class to
            register. If not specified, all magic commands are registered.

    """

    ip = get_ipython()
    if ip is None:
        raise RuntimeError('No iPython shell found')
    else:
        if magic_commands is not True:
            # filter out any magic commands that are not in magic_commands
            cls.magics = {line_cell: {key: val for key, val in magics.items()
                                      if key in magic_commands}
                          for line_cell, magics in cls.magics.items()}
        ip.magics_manager.register(cls)
