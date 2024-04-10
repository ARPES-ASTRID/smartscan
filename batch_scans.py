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

    Args:
        settings (dict | str): The settings dictionary or the path to the settings file.
    """
    logger = logging.getLogger("Batch Manager")
    if isinstance(settings, str):
        with open(settings, "r") as file:
            settings = yaml.safe_load(file)

    # ~~~ SETUP BATCHES BELOW ~~~
    i = 1
    tasks = {
        "laplace_filter": {
            "function": "laplace_filter",
            "params": {"sigma": 2, "roi": [[145, 190], [10, 140]]},
        },
        "curvature": {
            "function": "curvature",
            "params": {
                "bw": 5,
                "c1": 0.001,
                "c2": 0.001,
                "w": 1,
                "roi": [[145, 190], [10, 140]],
            },
        },
    }
    rois = [[[145, 190], [10, 140]], [[45, 190], [10, 140]]]

    for roi in rois:
        logger.info(f"Starting batch run #{i}")
        settings["scanning"]["n_points"] = 499
        settings["scanning"]["duration"] = 3600 / 2

        settings["cost_function"]["params"]["weight"] = 1
        settings["acquisition_function"]["params"]["a"] = 3

        settings["tasks"] = {
            "mean": {
                "function": "mean",
                "params": {
                    "roi": roi,
                },
            },
            "curvature": {
                "function": "curvature",
                "params": {"bw": 5, "c1": 0.001, "c2": 0.001, "w": 1, "roi": roi},
            },
        }

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run(settings)
        logger.info(f"Finished batch run #{i}")
        logger.info(f"Waiting for 30 seconds")
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f"Starting batch run #{i}")
        settings["scanning"]["n_points"] = 499
        settings["scanning"]["duration"] = 3600 / 2

        settings["cost_function"]["params"]["weight"] = 10
        settings["acquisition_function"]["params"]["a"] = 3

        settings["tasks"] = {
            "mean": {
                "function": "mean",
                "params": {
                    "roi": roi,
                },
            },
            "curvature": {
                "function": "curvature",
                "params": {"bw": 5, "c1": 0.001, "c2": 0.001, "w": 1, "roi": roi},
            },
        }

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run(settings)
        logger.info(f"Finished batch run #{i}")
        logger.info(f"Waiting for 30 seconds")
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f"Starting batch run #{i}")
        settings["scanning"]["n_points"] = 499
        settings["scanning"]["duration"] = 3600 / 2

        settings["cost_function"]["params"]["weight"] = 50
        settings["acquisition_function"]["params"]["a"] = 3

        settings["tasks"] = {
            "mean": {
                "function": "mean",
                "params": {
                    "roi": roi,
                },
            },
            "curvature": {
                "function": "curvature",
                "params": {"bw": 5, "c1": 0.001, "c2": 0.001, "w": 1, "roi": roi},
            },
        }

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run(settings)
        logger.info(f"Finished batch run #{i}")
        logger.info(f"Waiting for 30 seconds")
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f"Starting batch run #{i}")
        settings["scanning"]["n_points"] = 499
        settings["scanning"]["duration"] = 3600 / 2

        settings["cost_function"]["params"]["weight"] = 50
        settings["acquisition_function"]["params"]["a"] = 3

        settings["tasks"] = {
            "mean": {
                "function": "mean",
                "params": {
                    "roi": roi,
                },
            },
            "curvature": {
                "function": "curvature",
                "params": {"bw": 5, "c1": 0.001, "c2": 0.001, "w": 1, "roi": roi},
            },
        }

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run(settings)
        logger.info(f"Finished batch run #{i}")
        logger.info(f"Waiting for 30 seconds")
        time.sleep(60)
        i += 1

        # ~~~RUN ~~~
        logger.info(f"Starting batch run #{i}")
        settings["scanning"]["n_points"] = 499
        settings["scanning"]["duration"] = 3600 / 2

        settings["cost_function"]["params"]["weight"] = 100
        settings["acquisition_function"]["params"]["a"] = 3

        settings["tasks"] = {
            "mean": {
                "function": "mean",
                "params": {
                    "roi": roi,
                },
            },
            "curvature": {
                "function": "curvature",
                "params": {"bw": 5, "c1": 0.001, "c2": 0.001, "w": 1, "roi": roi},
            },
        }

        print(f"\n\n\n")
        logger.info(f"\ttasks: {settings['tasks'].keys()}")
        logger.info(f"\troi: {settings['tasks']['mean']['params']['roi']}")
        logger.info(f"\ta: {settings['acquisition_function']['params']['a']}")
        logger.info(f"\tcost weight: {settings['cost_function']['params']['weight']}")
        print("\n\n\n")
        run(settings)
        logger.info(f"Finished batch run #{i}")
        logger.info(f"Waiting for 30 seconds")
        time.sleep(60)
        i += 1


if __name__ == "__main__":
    main(batches)
