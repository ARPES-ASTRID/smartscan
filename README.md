# smartscan

Smartscan is a tool that allows autonomous control of nanoARPES experiments.
It uses Gaussian Process regression to estimate the the most promising next
point to measure in an n-dimensional parameter space. The tool is designed to
be used with the SGM4 endstation at the ASTRID2 synchrotron in Aarhus, Denmark,
but can be adapted to other endstations.

## Installation

To install the package, clone the repository and run the following command in
the root directory:

```bash
pip install -e .
```

## Usage

### Real End-station

Starting a new scan is done by running the `smartscan.py` script. The script will
load the settings from the `scan_config.json` file and start the scan. The scan
will run until the specified number of measurements have been taken.

### Simulation

To simulate running a scan without actually controlling any end-station, the
`simulator.py` script can be used. The script will load the settings from the
`simulator_config.json` and run as if it was controlling the end-station.

Once the simulator is running, the `smartscan.py` script can be run to start the
scan. The simulator will respond to the commands from the `smartscan.py` script as
if it was the real end-station.
