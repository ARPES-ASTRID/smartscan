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


def batches(settings,logger) -> None:
    roi_1 = [[50, 190], [10, 140]]
    roi_2 = [[50, 110], [10, 140]]
    for roi in [roi_1,roi_2]:
    # a_vals = [0.01,1]
        settings['scanning']['']
        tasks = {
            'laplace_filter': {
                'function': 'laplace_filter',
                'params':{
                    'sigma': 5,
                    'norm': False,
                    'roi': roi,
                    },
            },
            'contrast_noise_ratio': {
                'function': 'contrast_noise_ratio',
                'params': {
                    'signal_roi': roi,
                    'bg_roi': [[150,200], [50, 100]],
                },
            },
            'mean':{'function': 'mean', 'params': {'roi': roi,}},
            'std':{'function': 'std', 'params': {'roi': roi,}},
        }
        i = 1

        # ~~~batch~~~~
        settings['tasks'] = {tasks['mean'],tasks['laplace_filter']}
        settings['acquisition_function']['params']['a'] = 0.1
        settings['cost_function']['params']['weight'] = 0.01
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(30)
        i += 1
        # ~~~batch~~~~
        settings['tasks'] = {tasks['laplace_filter'],tasks['contrast_noise_ratio']}
        settings['acquisition_function']['params']['a'] = 0.1
        settings['cost_function']['params']['weight'] = 0.01
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(30)
        i += 1
        # ~~~batch~~~~
        settings['tasks'] = {tasks['std'],tasks['laplace_filter']}
        settings['acquisition_function']['params']['a'] = 0.1
        settings['cost_function']['params']['weight'] = 0.01
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(30)
        i += 1
        # ~~~batch~~~~
        settings['tasks'] = deepcopy(tasks)
        settings['gp']['fvgp']['init_hyperparameters'] = [1_000_000, 100, 100, 1]
        settings['gp']['training']['hyperparameter_bounds'] = [
            [1_000_000, 1_000_000_000],
            [10, 1000],
            [10, 1000],
            [0.001, 4]
            ]
        settings['acquisition_function']['params']['a'] = 0.1
        settings['cost_function']['params']['weight'] = 0.01
        tasks['contrast_noise_ratio']['params']['bg_roi'] = [[200,-1], [0, -1]]
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
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
