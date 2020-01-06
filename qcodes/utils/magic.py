import sys
from IPython.core.magic import Magics, magics_class, line_cell_magic, cell_magic
from IPython import get_ipython
import threading


from qcodes.utils.helpers import define_func_from_string

if sys.version_info < (3, 6):
    raise RuntimeError('Magic only supported for Python version 3.6 and up')


@magics_class
class QCoDeSMagic(Magics):
    """Magics related to code management (loading, saving, editing, ...)."""

    def __init__(self, *args, **kwargs):
        self._knowntemps = set()
        super(QCoDeSMagic, self).__init__(*args, **kwargs)

    @line_cell_magic
    def measurement(self, line, cell=None):
        """
        Create qcodes.Loop measurement mimicking Python `for` syntax via
        iPython magic.
        Upon execution of a notebook cell, the code is transformed from the
        for loop structure to a QCoDeS Loop before being executed.
        Can be run by having %%measurement in the first line of a cell,
        followed by the measurement name (see below for an example)

        The for loop syntax differs slightly from a Python `for` loop,
        as it uses `for {iterable}` instead of `for {element} in {iterable}`.
        The reason is that `{element}` cannot be accessed (yet) in QCoDeS loops.

        Comments (#) are ignored in the loop.
        Any code after the loop will also be run, if separated by a blank
        line from the loop.
        The Loop object is by default stored in a variable named `loop`,
        and the dataset in `data`, and these can be overridden using options.
        Must be run in a Jupyter Notebook.
        Delays can be provided in a loop by adding `-d {delay}` after `for`

        The following options can be passed along with the measurement name
        (e.g. %%measurement -px -d data_name {measurement_name}):
            -p : print transformed code
            -x : Do not execute code
            -d <dataset_name> : Use custom name for dataset
            -l <loop_name> : Use custom name for Loop

        An example for a loop cell is as follows:

        %%measurement {-options} {measurement_name}
        for {sweep_vals}:
            {measure_parameter1}
            {measure_parameter2}
            for -d 1 {sweep_vals2}:
                {measure_parameter3}

        {Additional code}
        ```

        which will be internally transformed to:

        ```
        import qcodes
        loop = qcodes.Loop({sweep_vals}).each(
            {measure_parameter1},
            {measure_parameter2},
            qcodes.Loop({sweep_vals2}, delay=1).each(
                {measure_parameter3}))
        data = loop.get_data_set(name={measurement_name})

        {Additional code}
        ```

        An explicit example of the line `for {sweep_vals}:` could be
        `for sweep_parameter.sweep(0, 42, step=1):`

        """

        if cell is None:
            # No loop provided, print documentation
            print(self.measurement.__doc__)
            return

        # Parse line, get measurement name and any possible options
        options, msmt_name = self.parse_options(line, 'psrd:l:x')
        data_name = options.get('d', 'data')
        loop_name = options.get('l', 'loop')

        lines = cell.splitlines()
        assert lines[0][:3] == 'for', "Measurement must start with for loop"

        contents = 'import qcodes\n{} = '.format(loop_name)
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
                    line_representation += '{},\n'.format(line)
                contents += line_representation

                # Remember level for next iteration (might exit inner loop)
                previous_level = level

        # Add closing brackets for any remaining loops
        contents += ')' * previous_level + '\n'

        if '{' in msmt_name:
            # msmt_name contains code to be executed (withing {} braces).
            # Usually it is executed before calling function, but if it
            # raises an error, it is passed here without being executed.
            # Could occur if msmt_name contains reference to loop

            # Execute loop since msmt_name code may refer to it
            exec(contents, self.shell.user_ns)
            import re
            keys = re.findall(r"\{([A-Za-z0-9\[\]\(\)\*.+-_]+)\}", msmt_name)
            for key in keys:
                if key in globals():
                    # variable exists in current namespace
                    val = eval(key)
                else:
                    # variable exists in user namespace
                    val = eval(key, self.shell.user_ns)
                msmt_name = msmt_name.replace(f'{{{key}}}', str(val), 1)

        # Add dataset
        contents += f"{data_name} = {loop_name}.get_data_set(name='{msmt_name}')"
        if 's' not in options:
            contents += f"\nprint({data_name})"

        for line in lines[k+1:]:
            contents += '\n' + line

        if 'r' in options:
            contents += f'\n{loop_name}.run(thread=True)'

        if 'p' in options:
            print(contents)

        if 'x' not in options:
            # Execute contents
            self.shell.run_cell(contents, store_history=True, silent=True)

    @cell_magic
    def measurement_job(self, line, cell):
        """Run a Measurement in a dedicated separate thread.
        All the code in the cell is considered part of the measurement and is
        therefore executed in a separate thread.
        The Measurement refers to silq.tools.loop_tools.Measurement
        The measurement thread is an IPython job. It can be accessed via
        `qcodes.measurement_job`, and the job manager via `qcodes.measurement_jobs`.

        Note that only one Measurement job can be active at a time.

        A side effect is that the code in the cell is first converted to a
        function whose name is _measurement

        Args:
            line: Optional name for the thread. Default is '_measurement'.
                If a different name is used, it will not update
                `qcodes.measurement_job`. This function can therefore also be
                used to run arbitrary code in a separate thread
            cell: Code to be executed in a separate thread

        Raises:
            RuntimeError if a measurement thread already exists

        Examples:
            the following code should be run in a cell:

            %%measurement_job
            p_sweep = Parameter('sweep_parameter', set_cmd=None)
            p_measure = Parameter('measurable', initial_value=0)
            with Measurement('my_msmt') as msmt:
                for sweep_val in p_sweep.sweep(0, 1, 11):
                    msmt.measure(p_measure)


        """
        thread_name = line or '_measurement'
        # TODO Incorporate line arguments to change following settings
        allow_duplicate_threads = False

        # Ensure thread does not already exist if allow_duplicate_threads = False
        if not allow_duplicate_threads:
            existing_thread_names = [thread.name for thread in threading.enumerate()]
            if thread_name in existing_thread_names:
                raise RuntimeError(f'Thread {thread_name} already exists. Exiting')

        # Create new function that will be passed to thread
        f = define_func_from_string(thread_name, cell)

        # Run thread as an IPython job, registered in the QCoDeS job manager
        import qcodes
        job = qcodes.measurement_jobs.new(f)

        # Register current job as active job
        if thread_name == '_measurement':
            qcodes.measurement_job = job


def register_magic_class(cls=QCoDeSMagic, magic_commands=True):
    """
    Registers a iPython magic class
    Args:
        cls: magic class to register. Default is QCoDeSMagic
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
