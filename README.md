
# smartscan

Smartscan is a tool that allows autonomous control of nanoARPES experiments.
It uses Gaussian Process regression to estimate the the most promising next
point to measure in an n-dimensional parameter space. The tool is designed to
be used with the SGM4 endstation at the ASTRID2 synchrotron in Aarhus, Denmark,
but can be adapted to other endstations.

The Gaussian Process regression is implemented using the `GPcam` package (v7.4.10), description of
which can be found [here](https://gpcam.lbl.gov/) and the related paper can be found [here](https://www.nature.com/articles/s42254-021-00345-y).

## Installation

To install the package, clone the repository and run the following command in
the root directory:

```bash
pip install -e .
```

Smartscan is built using Python 3.10 and has been tested on Windows only. It should work on other platforms as well, but this has not been tested. The package has the following dependencies:

*GPcam*:
  - gpcam == 7.4.10
  - fvgp == 3.3.8
  - hgdl == 2.0.7

*Other*:
numpy, scipy, h5py, matplotlib, xarray, tqdm.

*We strongly recommend using a virtual environment to install the package.*

If installing using the command above, no additional dependencies are needed, as they are
installed automatically. 
However, if you want to install the dependencies manually, the required packages are listed in the 
`env.yml` file. A manual installation of the dependencies using `conda` can be done by running the following
command:

```bash
conda env create -f env.yml
```

which will create a new environment called `smartscan` with all the necessary
dependencies.

## Usage

Here we describe how to use the `smartscan` tool to run a scan on the SGM4 endstation.
The tool can be used to run a scan on the real endstation, or to simulate a scan using
pre-measured data.

### SGM4 @ ASTRID2

To start a new scan, the SGM4 end station needs to be set up for a remote autonomous scan. 
Once that is running and listening on the TCP port, the GP driven scan can be started with
```bash
python -m smartscan
```
When no options are provided, the scan will be set up uning the settings from the 
`config.json`. These settings can be changed to any other file using the `--config` option.
```bash
python -m smartscan --config path/to/config.json
```

The scan will run until the specified number of measurements have been taken, or 
until the scan is stopped manually using `Ctrl+C`.

### Simulation

To simulate running a scan without actually controlling any end-station, a dummy end-station
can be started with
```bash
python -m smartscan.simulator
```
This dummy end-station will respond to the commands from the `smartscan` script as if it was
the real end-station. The data returned by the simulator is picked from a pre-measured raster map.
The file used for this is selected either in the configuration file or by providing the `--data` option.
```bash
python -m smartscan.simulator --data path/to/data.h5
```
Also here, the default settings for the simulator can be changed by providing a custom configuration 
file
```bash
python -m smartscan.simulator --config path/to/config.json
```
Once the simulator is running, a regular scan can be started as described above.

### Data for Simulation

The data used for the simulation is stored in an HDF5 file format designed for the SGM4 endstation.
Example datasets can can be found in the zenodo repository 
 <!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5550004.svg)](https://doi.org/10.5281/zenodo.5550004). -->

<!-- ## Configuration

The configuration file is a YAML file that contains all the settings for the scan.
The following settings are available: -->

# Citation

If you use this tool in your research, please cite the following paper:

```bibtex
@article{smartscan,
  title={Autonomous Micro-Focus Angle-Resolved Photoemission Spectroscopy},
  author={Steinn Ymir Agustsson, Alfred J.H.Jones, Davide Curcio, Søren Ulstrup, Jill Miwa, Davide Mottin, Panagiotis Karras and Philip Hofmann},
  journal={Review of Scientific Instruments},
  year={2024}
  status={In Review}
}
```
