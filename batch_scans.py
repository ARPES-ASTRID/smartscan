"""Script to run multiple scans in a batch."""

import logging
import time

import yaml

from smartscan.smartscan import main, run


def batches(settings: dict | str) -> None:
    """Run multiple scans in a batch.

    This function can be modified to run multiple scans in a batch. The settings
    and logger are passed to the function.
    Each batch is  started by calling the `run` function with the settings dictionary,
    modified for the specific batch.
    After each run is finished, the function waits for 60 seconds before starting the next.
    This ensures the system has enough time to reset and the data is saved properly.

    Args:
        settings (dict | str): The settings dictionary or the path to the settings file.
    """
    logger = logging.getLogger("Batch Manager")
    if isinstance(settings, str):
        with open(settings, "r") as file:
            settings = yaml.safe_load(file)
    i = 0
    # ~~~ SETUP BATCHES BELOW ~~~

    # ~~ BATCH ~~
    i += 1
    settings["cost_function"]["params"]["weight"] = 1
    settings["acquisition_function"]["params"]["a"] = 3
    run(settings)
    logger.info(f"Finished batch run #{i}")
    logger.info(f"Waiting for 60 seconds")
    time.sleep(60)

    # ~~ BATCH ~~
    i += 1
    settings["cost_function"]["params"]["weight"] = 1
    settings["acquisition_function"]["params"]["a"] = 1
    run(settings)
    logger.info(f"Finished batch run #{i}")
    logger.info(f"Waiting for 60 seconds")
    time.sleep(60)

    # ~~ BATCH ~~
    i += 1
    settings["cost_function"]["params"]["weight"] = 10
    settings["acquisition_function"]["params"]["a"] = 10
    run(settings)
    logger.info(f"Finished batch run #{i}")
    logger.info(f"Waiting for 60 seconds")
    time.sleep(60)


if __name__ == "__main__":
    main(batches)
