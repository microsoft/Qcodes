import logging
import os

_LOG = logging.getLogger(__name__)


def add_to_spyder_UMR_excludelist(modulename: str) -> None:
    """
    Spyder tries to reload any user module. This does not work well for
    qcodes because it overwrites Class variables. QCoDeS uses these to
    store global attributes such as default station, monitor and list of
    instruments. This "feature" can be disabled by the
    gui. Unfortunately this cannot be disabled in a natural way
    programmatically so in this hack we replace the global ``__umr__`` instance
    with a new one containing the module we want to exclude. This will do
    nothing if Spyder is not found.
    TODO is there a better way to detect if we are in spyder?
    """
    if any("SPYDER" in name for name in os.environ):
        sitecustomize_found = False
        try:
            from spyder.utils.site import (  # pyright: ignore[reportMissingImports]
                sitecustomize,
            )
        except ImportError:
            pass
        else:
            sitecustomize_found = True
        if sitecustomize_found is False:
            try:
                from spyder_kernels.customize import spydercustomize  # pyright: ignore

                sitecustomize = spydercustomize  # noqa F811
            except ImportError:
                pass
            else:
                sitecustomize_found = True

        if sitecustomize_found is False:
            return

        excludednamelist = os.environ.get("SPY_UMR_NAMELIST", "").split(",")
        if modulename not in excludednamelist:
            _LOG.info(f"adding {modulename} to excluded modules")
            excludednamelist.append(modulename)
            sitecustomize.__umr__ = sitecustomize.UserModuleReloader(  # pyright: ignore[reportPossiblyUnboundVariable]
                namelist=excludednamelist
            )
            os.environ["SPY_UMR_NAMELIST"] = ",".join(excludednamelist)
