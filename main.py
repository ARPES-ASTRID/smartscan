""" Main file to run the asyncscanner. """
import os
import sys
import logging
import argparse
import datetime
import numpy as np
import yaml

from pathlib import Path
from smartscan.utils import ColoredFormatter

def main_asyncio(settings) -> None:
    import asyncio
    from smartscan import AsyncScanManager

    # init scan manager
    scan_manager = AsyncScanManager(settings=parsed_args.settings, logger=logger)
    # start scan manager
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scan_manager.start())
    except KeyboardInterrupt:
        scan_manager.remote.END()
        logger.error('Terminated scan from keyboard')
    loop.close()
    logger.info("Scan manager stopped.")

def main_gui(settings) -> None:
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AsyncScanManager")
    parser.add_argument("--host", type=str, default="localhost", help="SGM4 host")
    parser.add_argument("--port", type=int, default=54333, help="SGM4 port")
    parser.add_argument(
        "--settings", type=str, default="scan_settings.yaml", help="Settings file"
    )
    parser.add_argument(
        "--gui", default= True,action="store_true", help="Start as GUI"
    )
    # parser.add_argument('--loglevel',   type=str, default='DEBUG', help='Log level')
    # parser.add_argument('--logdir',     type=str, default=None, help='Log directory')
    # parser.add_argument('--duration',   type=int, default=None, help='Duration of the scan in seconds')
    # parser.add_argument('--train_at',   type=int, default=None, help='Train GP at these number of samples')
    # parser.add_argument('--train_every',type=int, default=None, help='Train GP every n samples')
    parsed_args = parser.parse_args()

    # load settings from json file
    settings_file = parsed_args.settings
    with open(settings_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # init logger
    # logger.setLevel("DEBUG")#settings["logging"]["level"])
    logging.root.setLevel(settings["logging"]["level"])
    formatter = ColoredFormatter(settings["logging"]["formatter"])

    sh = logging.StreamHandler()
    sh.setLevel(settings["logging"]["level"])
    sh.setFormatter(formatter)
    logging.root.addHandler(sh)

    logger = logging.getLogger(__name__)

    if settings["logging"]["directory"] is not None:
        logdir = Path(settings["logging"]["directory"])
        if not logdir.exists():
            logdir.mkdir()
        # highest number of existing logfiles
        i = 0
        for file in logdir.iterdir():
            if file.suffix == ".txt":
                i = max(i, int(file.stem.split("_")[2]))

        # current datetime
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        logname = f"scan_log_{i}_{now}.txt"

        fh = logging.FileHandler(logdir / logname)
        fh.setLevel("DEBUG")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.debug('Created logger at level %s' % logger.level)


    # numpy compact printing
    np.set_printoptions(precision=3, suppress=True)

    # an unsafe, unsupported, undocumented workaround :(
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if parsed_args.gui:
        main_gui(settings=settings_file)
    else:
        main_asyncio(settings=settings_file)
