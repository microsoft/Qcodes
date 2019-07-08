import os
import subprocess
import sys
import shutil
import datetime
import time
import json
import warnings

import qcodes


def update_qcodes_installation(back_up=True, env_name='qcodes', env_backup_name='qcodes_backup',
                               conda_update=True, env_update=True, pip_upgrade=False, qcodes_update=True):

    """
    Update current QCoDeS installation.

    Args:
        back_up (bool): Back up current installation before update

        env_name (str): Name of the environment to be updated

        env_backup_name (str): Name of the back up environment to be created

        conda_update (bool): Update 'conda' manager of Anaconda Python installation

        env_update (bool): Update QCoDeS python environment

        pip_upgrade (bool): When True, upgrade 'pip' and all the packages in the environment
        listed as 'outdated' by pip. This argument should be used with care.

        qcodes_update (bool): Updates QCoDeS module

    Raises:
        ImportError: If QCoDeS module cannot be imported after the update
    """

    # Will back up the QCoDeS Anaconda Python environment.

    _source = os.sep.join(qcodes.__file__.split(os.sep)[:-2])

    if not back_up == False:

        _name = str(datetime.date.today()) # Current date.
        env_backup_name = env_backup_name + '_' + _name

        print("Existing {} environment will be backed up as {}...\n".format(env_name, env_backup_name))

        subprocess.run('conda create --name {} --clone {}'.format(env_backup_name, env_name), shell=True)

        # Copy QCoDeS root to qcodes_backup

        _destination = _source + '_backup' + '_' + _name

        shutil.copytree(_source, _destination, symlinks=True, ignore=None)

        # Re-install QCoDeS from the back up root

        subprocess.run('activate {} && pip uninstall qcodes && pip install -e {}'
                       .format(env_backup_name, _destination), shell=True)

    # Will update the Conda environment.

    if not conda_update == False:

        print("Updating Conda...\n")
        subprocess.run('conda update -n base conda -c defaults', shell=True)

    # Update QCoDeS Environment

    # Note that Conda will not upgrade a package, even if there is a newer
    # version, in the case that the package satisfies the requirement specified
    # by QCoDeS.

    if not env_update == False:

        print("Updating QCoDeS environment...\n")
        subprocess.run('cd {} && conda env update'.format(_source), shell=True)

    # Update pip

    if pip_upgrade == True:

        print("Upgrading pip...\n")
        subprocess.run('pip install --user --upgrade pip', shell=True)

        print("Querying outdated packages...\n")
        out_pkgs = subprocess.run('pip list --outdated --format=json', shell=False,
                                  check=False, stdout=subprocess.PIPE).stdout
        _out_pkgs = out_pkgs.decode('utf8').replace("'", '"')
        _out_pkgs_dict = json.loads(_out_pkgs)

        print("Upgrading packages...")

        for i in range(len(_out_pkgs_dict)):
            pkg = _out_pkgs_dict[i]["name"]
            if not (pkg == 'qcodes' and pkg == 'qdev_wrappers'):
                subprocess.run('pip install -U --upgrade-strategy=only-if-needed ' +
                               '{}'.format(pkg), shell=True)
            else:
                pass

    # Update QCoDeS

    if not qcodes_update == False:

        print('Making a pull request from QCoDeS repositories...\n')

        subprocess.run('cd {} && git stash && git checkout master && git pull && git stash pop'.format(_source), shell=True)

        print("Updating QCoDeS...\n")

        subprocess.run('pip install -e {}'.format(_source), shell=True)

    # Test the installation

        try:

            import qcodes as qc
            print("QCoDeS version {} is succesfully installed.".format(qc.__version__))
            sys.exit()

        except ImportError:

            warnings.warn("An unknown issue occured during update.\nThe changes shall be rolled back.", UserWarning, 2)

            subprocess.run('conda deactivate && conda remove --name qcodes --all', shell=True)

            print("Cloning QCoDeS from back up...\n")

            subprocess.run('conda create --name {} --clone {}'.format('qcodes', env_backup_name), shell=True)
            subprocess.run('conda remove --name {} --all'.format(env_backup_name), shell=True)

            # Roll back to the roots

            shutil.rmtree(_source)
            shutil.copytree(_destination, _source, symlinks=True, ignore=None)

            subprocess.run('activate qcodes && pip uninstall qcodes && pip install -e {}'
                           .format(_source), shell=True)

            shutil.rmtree(_destination)

            sys.exit()
