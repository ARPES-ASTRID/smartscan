""" Main file to run the asyncscanner. """
import os
import sys
import logging
import asyncio
import time
from pathlib import Path
import traceback
from copy import deepcopy

import numpy as np
import yaml

from smartscan.utils import ColoredFormatter
from smartscan import AsyncScanManager


batched = False

def batches(settings,logger):
    aqf_batch(settings,logger)

def aqf_batch(settings,logger) -> None:
    tasks = {
        'laplace_filter': {            
            'function': 'laplace_filter', 
            'params': {
                'sigma': 10, 
                'roi': [[145,190],[10,140]]
            }
        },
        'curvature': {
            'function': 'curvature', 
            'params': {
                'bw': 5,
                'c1': 0.001,
                'c2': 0.001,
                'w': 1,
                'roi': [[145, 190], [10, 140]]
            }
        },
    }
    rois = [[145,190],[10,140]], [[45,190],[10,140]]
    aqf_values = [0.1,0.5,1]
    i = 1
    # ~~~batch~~~~
    for task in ["curvature","laplace_filter"]:
        for roi in rois:
            settings['tasks'] = {
                "mean": {
                    "function": "mean",
                    "params": {
                        "roi":roi
                    }
                },
                task: tasks[task]
            }
            settings['tasks'][task]['params']['roi'] = roi
            for val in aqf_values:
                logger.info(f"Starting batch run #{i}")
                settings['acquisition_function']['params']['a'] = val
                print(f"tasks: {settings['tasks'].keys()}")
                print(f"roi: {settings['tasks']['mean']['params']['roi']}")
                print(f"a: {settings['acquisition_function']['params']['a']}")
                run_asyncio(settings)
                logger.info(f'Finished batch run #{i}')
                logger.info(f'Waiting for 30 seconds')
                time.sleep(30)
                i += 1
    




#############################################################################

def run_asyncio(settings) -> None:

    logger = logging.getLogger(__name__)
    # init scan manager
    scan_manager = AsyncScanManager(settings=settings)
    # start scan manager
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scan_manager.start())
    except KeyboardInterrupt:
        loop.run_forever()
        loop.run_until_complete(scan_manager.stop())
        logger.warning('Terminated scan from keyboard')
    except Exception as e:
        logger.critical(f'Scan manager stopped due to {type(e).__name__}: {e} {traceback.format_exc()}')
        logger.exception(e)
        loop.run_until_complete(scan_manager.stop())

    logger.info("Scan manager stopped.")
    logger.info('Scan finished')


def run_gui(settings) -> None:
    from smartscan.gui import SmartScanApp
    
    app = SmartScanApp(
        sys.argv, 
        settings=settings,
    )
    
    try:
        app.exec_()
    except:
        print('exiting')
    # sys.exit(app.exec_())


if __name__ == '__main__':

    settings_file = "scan_settings.yaml"

    with open(settings_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    logging.root.setLevel(settings["logging"]["level"].upper())
    formatter = ColoredFormatter(settings["logging"]["formatter"])

    sh = logging.StreamHandler()
    sh.setLevel(settings["logging"]["level"].upper())
    sh.setFormatter(formatter)
    logging.root.addHandler(sh)

    logger = logging.getLogger(__name__)

    # suppress user warnings
    import warnings
    warnings.simplefilter('ignore', UserWarning)

    # numpy compact printing
    np.set_printoptions(precision=3, suppress=True)

    # an unsafe, unsupported, undocumented workaround :(
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if batched:
        logger.info("runnung batches")
        batches(settings=settings, logger=logger)
    else:
        logger.info("runnung single scan")
        run_asyncio(settings)

    asyncio.get_event_loop().stop()
    logger.info("Closed event loop")
