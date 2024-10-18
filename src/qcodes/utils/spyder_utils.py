import logging
import os
from packaging.version import Version


_LOG = logging.getLogger(__name__)


def add_to_spyder_UMR_excludelist(modulename: str) -> None:
    """
    Spyder tries to reload any user module. This does not work well for
    qcodes because it overwrites Class variables. QCoDeS uses these to
    store global attributes such as default station, monitor and list of
    instruments. This "feature" can be disabled by the
    gui. Unfortunately this cannot be disabled in a natural way
    programmatically so in this hack we retrieve the UMR instance
    and add the module we want to exclude. This will do
    nothing if Spyder is not found.
    TODO is there a better way to detect if we are in spyder?
    """
    if any("SPYDER" in name for name in os.environ):
        try:
            import spyder_kernels  # pyright: ignore

            if Version(spyder_kernels.__version__) < Version("3.0.0"):
                # In Spyder 4 and 5 UMR is a variable in module spydercustomize
                try:
                    from spyder_kernels.customize import spydercustomize  # pyright: ignore
                except ImportError:
                    return
                else:
                    umr = spydercustomize.__umr__
            else:
                # In Spyder 6 UMR is an attribute of the SpyderCodeRunner object.
                # This object can be found via the magics manager
                from IPython import get_ipython  # pyright: ignore
                ipython = get_ipython()
                if ipython is None:
                    # no ipython environment
                    return
                try:
                    runfile_method = ipython.magics_manager.magics['line']['runfile']
                except KeyError:
                    # no runfile magic
                    return
                spyder_code_runner = runfile_method.__self__
                umr = spyder_code_runner.umr

            excludednamelist = os.environ.get("SPY_UMR_NAMELIST", "").split(",")
            if modulename not in excludednamelist:
                _LOG.info(f"adding {modulename} to excluded modules")
                excludednamelist.append(modulename)
                if modulename not in umr.namelist:
                    umr.namelist.append(modulename)
                os.environ["SPY_UMR_NAMELIST"] = ",".join(excludednamelist)
        except Exception as ex:
            _LOG.warning(f"Failed to add {modulename} to UMR exclude list. {type(ex)}: {ex}")

