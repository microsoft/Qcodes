from qcodes.utils.natalie_wrappers.file_setup import CURRENT_EXPERIMENT
import os
import logging

log = logging.getLogger(__name__)


def save_device_image(sweeptparameters):
    counter = CURRENT_EXPERIMENT['provider'].counter
    title = "{} #{:03d}".format(CURRENT_EXPERIMENT["sample_name"], counter)
    di = CURRENT_EXPERIMENT['device_image']
    status = True
    if di.filename is None:
        status = di.loadAnnotations()

    if not status:
        log.warning("Could not load deviceannotation from disk. "
                    "No device image with be genereted for this "
                    "run")
        return
    di.updateValues(CURRENT_EXPERIMENT['station'], sweeptparameters)

    log.debug(os.path.join(CURRENT_EXPERIMENT["exp_folder"],
                           '{:03d}'.format(counter)))

    di.makePNG(CURRENT_EXPERIMENT["provider"].counter,
               os.path.join(CURRENT_EXPERIMENT["exp_folder"],
                            '{:03d}'.format(counter)), title)
