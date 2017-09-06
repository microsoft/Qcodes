from IPython.core.magic import Magics, magics_class, line_cell_magic


@magics_class
class QCoDeSMagic(Magics):
    """Magics related to code management (loading, saving, editing, ...)."""

    def __init__(self, *args, **kwargs):
        self._knowntemps = set()
        super(QCoDeSMagic, self).__init__(*args, **kwargs)

    @line_cell_magic
    def measurement(self, line, cell=None):
        """
        Create qcodes.Loop measurement using Python for syntax using iPython
        magic.
        Upon execution of a notebook cell, the code is transformed from the
        for loop structure to a QCoDeS Loop before being executed.
        Can be run by having %%measurement in the first line of a cell,
        followed by the measurement name (see below for an example)


        Comments (#) are ignored in the loop.
        Any code after the loop will also be run, if separated by a blank
        line from the loop.
        The Loop object is by default stored in a variable named `loop`,
        and the dataset in `data`, and these can be overridden using options.
        Must be run in a Jupyter Notebook.

        The following options can be passed along with the measurement name
        (e.g. %%measurement -px -d data_name {measurement_name}):
            -p : print transformed code
            -x : Do not execute code
            -d <dataset_name> : Use custom name for dataset
            -l <loop_name> : Use custom name for Loop

        An example for loop cell is:

        ```
        %%measurement {-options} {measurement_name}
        for {sweep_vals}:
            {measure_parameter1}
            {measure_parameter2}
            for {sweep_vals2}:
                {measure_parameter3}

        {Additional code}
        ```

        which will be internally transformed to:

        ```
        import qcodes
        loop = qcodes.Loop({sweep_vals}).each(
            {measure_parameter1},
            {measure_parameter2},
            qcodes.Loop({sweep_vals2}).each(
                {measure_parameter3}))
        data = loop.get_data_set(name={measurement_name})

        {Additional code}
        ```
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
                if level < previous_level :
                    # Exiting inner loop, close bracket
                    line_representation += '),' * (previous_level - level)
                    line_representation += '\n' + ' ' * level * 4

                if line[:3] == 'for':
                    # New loop
                    line_representation += f'qcodes.Loop({line[4:-1]}).each(\n'
                else:
                    # Action in current loop
                    line_representation += f'{line},\n'
                contents += line_representation

                # Remember level for next iteration (might exit inner loop)
                previous_level = level

        # Add closing brackets for any remaining loops
        contents += ')' * previous_level + '\n'
        # Add dataset
        contents += f"{data_name} = loop.get_data_set(name='{msmt_name}')"

        for line in lines[k+1:]:
            contents += '\n' + line

        if 'p' in options:
            print(contents)

        if 'x' not in options:
            # Execute contents
            self.shell.run_cell(contents, store_history=True, silent=True)


def register_magic_class(cls=QCoDeSMagic, magic_commands=True):
    """
    Registers a iPython magic class
    Args:
        cls: magic class to register
        magic_commands (List): list of magic commands within the class to
            register. If not specified, all magic commands are registered

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