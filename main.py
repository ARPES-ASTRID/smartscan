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
from smartscan.gp import cost_functions

from smartscan.utils import ColoredFormatter
from smartscan import AsyncScanManager


batched = True

def batches(settings,logger):
    i=1

    # ~~~ SETUP ~~~
    tasks = {
        'laplace_filter': {            
            'function': 'laplace_filter', 
            'params': {
                'sigma': 2,
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
    rois = [
        [[145,190],[10,140]], 
        [[45,190],[10,140]]
    ]

    for roi in rois:
        logger.info(f'Starting batch run #{i}')
        settings['scanning']['n_points'] = 499
        settings['scanning']['duration'] = 3600/2

        settings['cost_function']['params']['weight'] = 1
        settings['acquisition_function']['params']['a'] = 3

        settings['tasks'] = {
            "mean": {"function": "mean","params": {"roi":roi,}},
            'curvature': {'function': 'curvature','params': {
                    'bw': 5,'c1': 0.001,'c2': 0.001,'w': 1,
                    'roi': roi
                }}}

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(60)
        i += 1
            
        # ~~~RUN ~~~
        logger.info(f'Starting batch run #{i}')
        settings['scanning']['n_points'] = 499
        settings['scanning']['duration'] = 3600/2

        settings['cost_function']['params']['weight'] = 10
        settings['acquisition_function']['params']['a'] = 3

        settings['tasks'] = {
            "mean": {"function": "mean","params": {"roi":roi,}},
            'curvature': {'function': 'curvature','params': {
                    'bw': 5,'c1': 0.001,'c2': 0.001,'w': 1,
                    'roi': roi
                }}}

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f'Starting batch run #{i}')
        settings['scanning']['n_points'] = 499
        settings['scanning']['duration'] = 3600/2

        settings['cost_function']['params']['weight'] = 50
        settings['acquisition_function']['params']['a'] = 3

        settings['tasks'] = {
            "mean": {"function": "mean","params": {"roi":roi,}},
            'curvature': {'function': 'curvature','params': {
                    'bw': 5,'c1': 0.001,'c2': 0.001,'w': 1,
                    'roi': roi
                }}}

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f'Starting batch run #{i}')
        settings['scanning']['n_points'] = 499
        settings['scanning']['duration'] = 3600/2

        settings['cost_function']['params']['weight'] = 50
        settings['acquisition_function']['params']['a'] = 3

        settings['tasks'] = {
            "mean": {"function": "mean","params": {"roi":roi,}},
            'curvature': {'function': 'curvature','params': {
                    'bw': 5,'c1': 0.001,'c2': 0.001,'w': 1,
                    'roi': roi
                }}}

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f'Starting batch run #{i}')
        settings['scanning']['n_points'] = 499
        settings['scanning']['duration'] = 3600/2

        settings['cost_function']['params']['weight'] = 100
        settings['acquisition_function']['params']['a'] = 3

        settings['tasks'] = {
            "mean": {"function": "mean","params": {"roi":roi,}},
            'curvature': {'function': 'curvature','params': {
                    'bw': 5,'c1': 0.001,'c2': 0.001,'w': 1,
                    'roi': roi
                }}}

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run_asyncio(settings)
        logger.info(f'Finished batch run #{i}')
        logger.info(f'Waiting for 30 seconds')
        time.sleep(60)
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
