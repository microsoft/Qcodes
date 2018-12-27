# Generate large version 2 db files for benchmarking

import argparse
import os
import numpy as np

# NB: it's important that we do not import anything from qcodes before we
# do the git magic (which we do below), hence the relative import here
import utils as utils

parser = argparse.ArgumentParser(description='Generate large db file')
parser.add_argument('N', metavar='N', type=int,
                    help='number of runs to put into db file')

args = parser.parse_args()
N = args.N

fixturepath = os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2])
fixturepath = os.path.join(fixturepath, 'for_timing', 'db_files')


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
    path = os.path.join(v2fixturepath, f'holding_{N:06.0f}_runs.db')

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

    for _ in range(N):

        with meas.run() as datasaver:

            for x in np.random.rand(10):
                for y in np.random.rand(10):
                    z = np.random.rand()
                    datasaver.add_result((params[2], x),
                                         (params[3], y),
                                         (params[4], z))


if __name__ == '__main__':

    gens = (generate_DB_file_with_some_runs,)

    # pylint: disable=E1101
    utils.checkout_to_old_version_and_run_generators(version=2, gens=gens)
