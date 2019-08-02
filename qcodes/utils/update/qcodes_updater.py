from typing import Tuple

import datetime
import git
import json
import os
import shutil
import subprocess
import urllib.request
import warnings
import qcodes

class DirtyTreeException(Exception):
    pass


def address_keeper() -> Tuple[str, str]:
    # QCoDeS source path
    src = os.sep.join(qcodes.__file__.split(os.sep)[:-2])
    time_stamp = str(datetime.date.today()) # Current date.
    dst = src + '_backup' + '_' + time_stamp
    return src, dst


def qcodes_is_editable() -> bool:
    """
    Checks if QCoDeS installation is editable.
    """
    # Get editable installs
    edit_pkgs = subprocess.run('pip list -e --format=json', shell=False,
                                check=False, stdout=subprocess.PIPE).stdout
    edit_pkgs_ = edit_pkgs.decode('utf8').replace("'", '"')
    edit_pkgs_dict_ = json.loads(edit_pkgs_)
    for _, pkg_ in enumerate(edit_pkgs_dict_):
        pkg = pkg_['name']
        if pkg == 'qcodes':
            return True
    return False


def qcodes_backup(env_name: str = 'qcodes',
                  env_backup_name: str = 'qcodes_backup') -> None:
    """
    Back up QCoDeS Anaconda Python environment.

    Args:
        env_name: Name of the environment to be updated
        env_backup_name: Name of the back up environment to be created

    Raises:
        DirtyTreeException: If working tree of an editable install is dirty
    """
    time_stamp = str(datetime.date.today()) # Current date.
    env_backup_name = env_backup_name + '_' + time_stamp
    if qcodes_is_editable():
        source, destination = address_keeper()
        # Make sure that git working tree is clean
        repo = git.Repo(source)
        if repo.is_dirty():
            raise DirtyTreeException("QCoDeS working tree is not clean. " +
                                     "Please commit or stash your changes " +
                                     "and try again.")
        else:
            print(f'Existing {env_name} environment will be backed up as{env_backup_name}...\n')
            subprocess.run(f'conda create --name {env_backup_name} --clone {env_name}', shell=True)
            # Copy QCoDeS root to qcodes_backup
            shutil.copytree(source, destination, symlinks=True, ignore=None)
            # Re-install QCoDeS from the back up root
            subprocess.run(f'activate {env_backup_name} && pip uninstall qcodes && pip install -e {destination}', shell=True)
    else:
        print(f"Existing {env_name} environment will be backed up as {env_backup_name}...\n")
        subprocess.run(f'conda create --name {env_backup_name} --clone {env_name}', shell=True)


def conda_env_update(env_name: str = 'qcodes', env_backup_name: str = 'qcodes_backup',
                     conda_update: bool = True, env_update: bool = True,
                     back_up: bool = True) -> None:
    """
    Update Conda package manager and QCoDeS Anaconda Python environment.
    Note that Conda will not upgrade a package, even if there is a newer
    version, in the case that the package satisfies the requirement specified
    by QCoDeS.

    Args:
        env_name: Name of the environment to be updated
        env_backup_name: Name of the back up environment to be created
        conda_update: Update Conda pakage manager
        env_update: Update Anaconda Python environment
        back_up: Back up QCoDeS installation

    Raises:
        DirtyTreeException: If working tree of an editable install is dirty
    """
    source, _ = address_keeper()
    def execute_update() -> None:
        # Update Conda
        if conda_update:
            print("Updating Conda...\n")
            subprocess.run('conda update -n base conda -c defaults', shell=True)
        if env_update and qcodes_is_editable():
            # Pull QCoDeS master
            repo = git.Repo(source)
            _git = repo.git
            _git.checkout('master')
            _git.pull()
            # Update QCoDeS Anaconda Python environment
            print("Updating QCoDeS environment...\n")
            subprocess.run('conda env update --file environment.yml', shell=True, cwd=source)
        elif env_update:
            print("Updating QCoDeS environment...\n")
            file_name = 'environment.yml'
            url = 'https://raw.githubusercontent.com/QCoDeS/Qcodes/master/environment.yml'
            with urllib.request.urlopen(url) as r, open(file_name, 'wb') as f:
                shutil.copyfileobj(r, f)
                f.close()
                subprocess.run('conda env update --file environment.yml', shell=True, cwd=os.getcwd())
                os.remove(file_name)
    try:
        if back_up:
            qcodes_backup(env_name, env_backup_name)
            execute_update()
        else:
            execute_update()
    except DirtyTreeException:
        print("QCoDeS working tree is not clean. Please commit or stash your changes, and try again.")


def update_qcodes_installation(env_name: str = 'qcodes',
                               env_backup_name: str = 'qcodes_backup') -> None:
    """
    Update current QCoDeS Anaconda Python installation.

    Args:
        env_name: Name of the environment to be updated
        env_backup_name: Name of the back up environment to be created

    Raises:
        ImportError: If QCoDeS module cannot be imported after the update
        DirtyTreeException: If working tree of an editable install is dirty
    """
    source, destination = address_keeper()
    try:
        # Backup first
        qcodes_backup(env_name, env_backup_name)
        # Now update environment
        conda_env_update(env_name, env_backup_name, back_up=False)
        # Update QCoDeS
        if qcodes_is_editable():
            print('Updating QCoDeS from master...\n')
            subprocess.run(f'pip install -e {source}', shell=True)
        else:
            print('Updating QCoDeS via pip...\n')
            subprocess.run('pip install qcodes --upgrade', shell=True)
    except DirtyTreeException:
        print("QCoDeS working tree is not clean. Please commit or stash your changes, and try again.")
    except ImportError:
        warnings.warn("An unknown issue occured during update.\nThe changes shall be rolled back.", UserWarning, 2)
        subprocess.run('conda deactivate && conda remove --name qcodes --all', shell=True)
        print("Cloning QCoDeS from back up...\n")
        time_stamp = str(datetime.date.today()) # Current date.
        env_backup_name = env_backup_name + '_' + time_stamp
        subprocess.run(f'conda create --name {env_name} --clone {env_backup_name}', shell=True)
        subprocess.run(f'conda remove --name {env_backup_name} --all', shell=True)
        # To check whether we backed up an editable install, as the import fails,
        # we can not use 'qcodes_is_editable()' function. Instead, a simple
        # querry about the existance of the 'destination' directory should suffice
        # at this stage: i.e., if it exists, then it is an exact replica of
        # the QCoDeS local repository where the installation has been done.
        if os.path.isdir(destination):
            # Roll back to the roots
            shutil.rmtree(source)
            shutil.copytree(destination, source, symlinks=True, ignore=None)
            subprocess.run(f'activate qcodes && pip uninstall qcodes && pip install -e {source}', shell=True)
            shutil.rmtree(destination)
