# Generate version 2 database files for qcodes' test suite to consume

import os
import numpy as np

# NB: it's important that we do not import anything from qcodes before we
# do the git magic (which we do below), hence the relative import here
import utils as utils


fixturepath = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2])
fixturepath = os.path.join(fixturepath, 'fixtures', 'db_files')


def generate_empty_DB_file():
    """
    Generate the bare minimal DB file with no runs
    """

    import qcodes.dataset.sqlite_base as sqlite_base

    v2fixturepath = os.path.join(fixturepath, 'version2')
    os.makedirs(v2fixturepath, exist_ok=True)
    path = os.path.join(v2fixturepath, 'empty.db')

    if os.path.exists(path):
        os.remove(path)

    sqlite_base.connect(path)


def generate_DB_file_with_some_runs():
    """
    Generate a .db-file with a handful of runs with some interdependent
    parameters
    """

    # This function will run often on CI and re-generate the .db-files
    # That should ideally be a deterministic action
    # (although this hopefully plays no role)
    np.random.seed(0)

    v2fixturepath = os.path.join(fixturepath, 'version2')
    os.makedirs(v2fixturepath, exist_ok=True)
    path = os.path.join(v2fixturepath, 'some_runs.db')

    if os.path.exists(path):
        os.remove(path)

    from qcodes.dataset.sqlite_base import connect
    from qcodes.dataset.measurements import Measurement
    from qcodes.dataset.experiment_container import Experiment
    from qcodes import Parameter

    connect(path)
    exp = Experiment(path)
    exp._new(name='experiment_1', sample_name='no_sample_1')

    # Now make some parameters to use in measurements
    params = []
    for n in range(5):
        params.append(Parameter(f'p{n}', label=f'Parameter {n}',
                                unit=f'unit {n}', set_cmd=None, get_cmd=None))

    # Set up an experiment

    meas = Measurement(exp)
    meas.register_parameter(params[0])
    meas.register_parameter(params[1])
    meas.register_parameter(params[2], basis=(params[0],))
    meas.register_parameter(params[3], basis=(params[1],))
    meas.register_parameter(params[4], setpoints=(params[2], params[3]))

    # Make a number of identical runs

    for _ in range(10):

        with meas.run() as datasaver:

            for x in np.random.rand(10):
                for y in np.random.rand(10):
                    z = np.random.rand()
                    datasaver.add_result((params[2], x),
                                         (params[3], y),
                                         (params[4], z))


def generate_DB_file_with_empty_runs():
    """
    Generate a DB file that holds empty runs and runs with no interdependencies
    """

    v2fixturepath = os.path.join(fixturepath, 'version2')
    os.makedirs(v2fixturepath, exist_ok=True)
    path = os.path.join(v2fixturepath, 'empty_runs.db')

    if os.path.exists(path):
        os.remove(path)

    from qcodes.dataset.sqlite_base import connect
    from qcodes.dataset.measurements import Measurement
    from qcodes.dataset.experiment_container import Experiment
    from qcodes import Parameter
    from qcodes.dataset.data_set import DataSet

    conn = connect(path)
    exp = Experiment(path)
    exp._new(name='experiment_1', sample_name='no_sample_1')

    # Now make some parameters to use in measurements
    params = []
    for n in range(5):
        params.append(Parameter(f'p{n}', label=f'Parameter {n}',
                                unit=f'unit {n}', set_cmd=None, get_cmd=None))

    # truly empty run, no layouts table, no nothing
    dataset = DataSet(path, conn=conn)
    dataset._new('empty_dataset', exp_id=exp.exp_id)

    # empty run
    meas = Measurement(exp)
    with meas.run() as datasaver:
        pass

    # run with no interdeps
    meas = Measurement(exp)
    for param in params:
        meas.register_parameter(param)

    with meas.run() as datasaver:
        pass

    with meas.run() as datasaver:
        for _ in range(10):
            res = tuple((p, 0.0) for p in params)
            datasaver.add_result(*res)


if __name__ == '__main__':

    gens = (generate_empty_DB_file,
            generate_DB_file_with_some_runs,
            generate_DB_file_with_empty_runs)

    # pylint: disable=E1101
    utils.checkout_to_old_version_and_run_generators(version=2, gens=gens)
