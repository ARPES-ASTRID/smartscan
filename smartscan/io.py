from pathlib import Path
from typing import tuple

import dask.array
import h5py
import numpy as np
import xarray as xr
import yaml
from tqdm.auto import tqdm

from . import tasks


def load_smartscan(
    filename: str,
    folder: str | Path = None,
) -> dict["str", np.ndarray | xr.DataArray]:
    """Load data acquired with the GP driven smart scan on SGM4

    Loads and gathers the data from the h5 file and the settings from the yaml file. The data is
    returned as a list of xarray.DataArray, one for each position, and the settings are returned as
    a dictionary.

    Args:
        filename (str): file name
        folder (str|Path, optional): folder path. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: positions, data
    """
    h5_file_name = Path(filename).with_suffix(".h5")
    if not h5_file_name.exists():
        h5_file_name = Path(folder) / h5_file_name
    if not h5_file_name.exists():
        raise FileNotFoundError(f"{h5_file_name} does not exist")

    with h5py.File(h5_file_name, "r", swmr=True) as file:
        positions = file["Entry/Data/ScanDetails/TruePositions"][()]
        spectra = file["Entry/Data/TransformedData"][()]
        fa_len = file["Entry/Data/ScanDetails/FastAxis_length"][()]
        fa_start = file["Entry/Data/ScanDetails/FastAxis_start"][()]
        fa_step = file["Entry/Data/ScanDetails/FastAxis_step"][()]
        fa_name = file["Entry/Data/ScanDetails/FastAxis_names"][()].astype(str)
        sa_name = file["Entry/Data/ScanDetails/SlowAxis_names"][()].astype(str)
        coords = {}
        dims = list(sa_name) + list(fa_name)
        for i, name in enumerate(fa_name):
            start = fa_start[i]
            stop = fa_start[i] + fa_len[i] * fa_step[i]
            coords[name] = np.linspace(start, stop, fa_len[i])
        for i, name in enumerate(sa_name):
            coords[name] = None
    print(f"loaded {filename}: data shape: {spectra.shape}")

    # add settings
    settings_file = Path(filename + "_settings.yaml")
    if not settings_file.exists():
        settings_file = Path(folder) / (filename + ".yaml")
    if not settings_file.exists():
        settings_file = Path(folder) / (filename + "_settings.yaml")
    if not settings_file.exists():
        print(f"WARNING: settings file {settings_file} does not exist")
        settings = {}
    else:
        with open(settings_file) as file:
            settings: dict = get_scan_params(yaml.unsafe_load(file))
    print(f"loaded {settings_file}")

    out = {
        "all_positions": positions,
        "all_spectra": [],
        "dims": sa_name,
        "coords": coords,
        "attrs": {f"settings/{k}": v for k, v in settings.items()},
    }
    if len(settings) > 0:
        roi = out["attrs"]["settings/roi"]
        out["roi_dict"] = {k: slice(*v) for k, v in zip(fa_name, roi)}

    scan_attrs = get_h5_attrs(h5_file_name)
    out["attrs"].update(scan_attrs)

    for sp, pos in zip(spectra, positions):
        dims = list(sa_name) + list(fa_name)
        coords.update(
            {ax_name: np.array([pos[i]]) for i, ax_name in enumerate(sa_name)}
        )
        shape = [1] * len(sa_name) + list(sp.shape)
        # print(sp.shape, [f'{k}: {v.shape}' for k,v in coords.items()])
        out["all_spectra"].append(
            xr.DataArray(
                data=sp.reshape(shape),
                dims=dims,
                coords=coords,
                attrs=out["attrs"],
            ).squeeze()
        )
    out["all_spectra"] = xr.concat(out["all_spectra"], dim="idx")
    out["all_spectra"] = out["all_spectra"].assign_coords(
        idx=np.arange(len(out["all_spectra"].idx)),
    )

    merged = {}
    counts = {}
    # combine data with the same position
    for pos, sp in zip(out["all_positions"], out["all_spectra"]):
        pos = tuple(pos)
        if pos in merged:
            merged[pos] += sp
            counts[pos] += 1
        else:
            merged[pos] = sp
            counts[pos] = 1
    # get the mean of data with the same position
    merged = {k: v / counts[k] for k, v in merged.items()}
    out["unique_positions"] = np.array(tuple(merged.keys()))
    out["unique_counts"] = np.array(tuple(counts.values()))
    out["unique_spectra"] = xr.concat(merged.values(), dim="uidx")
    out["unique_spectra"] = out["unique_spectra"].assign_coords(
        uidx=np.arange(len(out["unique_spectra"].uidx))
    )

    # tasks
    if settings_file.exists():
        settings = yaml.unsafe_load(open(settings_file))
        out["tasks"] = {}
        for task, td in settings["tasks"].items():
            out["tasks"][task] = {
                "callable": getattr(tasks, td["function"]),
                "pars": td["params"],
            }
        out["task_values"] = []
    for i in tqdm(range(len(out["unique_spectra"]))):
        tvals = []
        sp = out["unique_spectra"].isel(
            uidx=i
        )  # .isel(out['roi_dict']).squeeze().values,
        for task, td in out["tasks"].items():
            tvals.append(td["callable"](sp, **td["pars"]))
        out["task_values"].append(tvals)
    out["task_values"] = np.array(out["task_values"])
    out["task_values_norm"] = out["task_values"] / out["task_values"].max(axis=0)
    out["task_values_norm"] = out["task_values"] / out["task_values"].max(axis=0)

    return out


def save_h5(
    xarr: xr.DataArray,
    filename: str,
    folder: str | Path = None,
    chunks: tuple | bool = False,
    compression: str = None,  #'gzip',
    mode: str = "w",
) -> None:
    """save an xarray.DataArray to h5 file

    This is a simpler alternative to the inbuilt NetCDF format. It saves the data in a more
    readable format, with the coordinates and dimensions saved as separate datasets.

    Args:
        xarr (xr.DataArray): xarray to save
        filename (str): file name
        folder (str|Path, optional): folder path. Defaults to None.
        chunks (tuple | bool, optional): chunk size. Defaults to False.
        compression (str, optional): compression type. Defaults to None.
        mode (str, optional): file mode. Defaults to 'w'.

    Raises:
        FileExistsError: if file exists and mode is 'w'
        FileNotFoundError: if file does not exist and mode is 'a'
        ValueError: if mode is not 'w' or 'a'
    """
    file_path = (Path(folder) / filename).with_suffix(".h5")
    if file_path.exists() and mode == "w":
        raise FileExistsError(f"{file_path} already exists")
    elif file_path.exists() and mode == "a":
        print(f"{file_path} already exists, appending")
    elif not file_path.exists() and mode == "a":
        raise FileNotFoundError(f"{file_path} does not exist")
    elif not file_path.exists() and mode == "w":
        print(f"{file_path} does not exist, creating")
    else:
        raise ValueError(f"invalid mode: {mode}")

    with h5py.File(file_path, mode=mode) as file:
        coord_group = file.create_group("coords")
        for k, v in xarr.coords.items():
            coord_group.create_dataset(f"{k}", data=v.values)
        dims = [str(d) for d in xarr.dims]
        file.create_dataset("dims", data=dims)
        file.create_dataset(
            "data", data=xarr.values, chunks=chunks, compression=compression
        )

    print(f"saved {filename} to {file_path}")


def load_h5(
    filename: str,
    folder: str | Path = None,
    lazy: bool = False,
) -> xr.DataArray:
    """Load data from h5 file to xarray.

    Loads data as saved by `save_h5`.

    Args:
        filename (str): file name
        folder (str|Path, optional): folder path. Defaults to None.
        lazy (bool, optional): lazy load using dask. Defaults to False.

    Returns:
        xr.DataArray: xarray
    """
    if folder is None:
        file_path = Path(filename).with_suffix(".h5")
    else:
        file_path = (Path(folder) / filename).with_suffix(".h5")
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")
    with h5py.File(file_path, "r") as file:
        dims = file["dims"][()].astype(str)
        coords = {}
        for k in file["coords"].keys():
            coords[k] = file["coords"][k][()]
        data = file["data"]
        if lazy:
            data = dask.array.from_(data, chunks=data.chunks)
        else:
            data = data[()]
    xarr = xr.DataArray(data, dims=dims, coords=coords)
    # print(f'loaded {filename}: data shape: {xarr.shape}')
    return xarr


def get_scan_params(sd: dict) -> dict:
    """Return a dict with the scan parameters

    Parse the scan dictionary and return a dictionary with the scan parameters, ready to be saved
    to an h5 file.

    Args:
        sd: the scan dictionary. Typically the result of loading the yaml file.

    Returns:
        dict: the scan parameters
    """
    params = {}
    params["max_points"] = sd["scanning"]["max_points"]
    params["duration"] = sd["scanning"]["duration"]
    params["normalize_values"] = sd["scanning"]["normalize_values"]
    try:
        roi_pars = sd["preprocessing"].get("roi", {}).get("params", {})
        if len(roi_pars) > 0:
            params["roi"] = list(roi_pars.values())
        else:
            params["roi"] = None
    except:  # noqa: E722
        params["roi"] = None
    # tasks
    for task, td in sd["tasks"].items():
        params["task_name"] = task
        if task == "laplace_filter":
            params["lap_sigma"] = td["params"]["sigma"]
            if "roi" in td["params"]:
                params["roi"] = td["params"]["roi"]
        else:
            params["lap_sigma"] = None
        if task == "mean":
            if td["params"] is not None and "roi" in td["params"]:
                params["roi"] = td["params"]["roi"]
        if task == "constrast_noise_ratio":
            params["roi"] = td["params"]["signal_roi"]
            params["cnr_bg_roi"] = td["params"]["bg_roi"]
        else:
            params["cnr_bg_roi"] = None
        if task == "std" and td["params"] is not None:
            params["std_roi"] = td["params"]["roi"]
        if task == "curvature":
            params["roi"] = td["params"]["roi"]

    # aquisition function
    params["acquisition_function"] = sd["acquisition_function"]["function"]
    for k, v in sd["acquisition_function"]["params"].items():
        params[f"aqf_{k}"] = v

    # cost function
    params["cost_function"] = sd["cost_function"]["function"]
    # params[f'min_distance'] = sd['cost_function']['params'].get('min_distance', 0)
    params["cf_speed"] = sd["cost_function"]["params"].get("speed", 300)
    params["cf_weight"] = sd["cost_function"]["params"].get("weight", None)

    return params


def reformat(v) -> float | str:
    """Reformat to numerical values the values from the h5 file when possible

    Args:
        v: the value to reformat

    Returns:
        float|str: the reformatted value
    """
    try:
        return float(v)
    except:  # noqa: E722
        return str(v)


def parse_h5_keys(
    d: str,
    prefix: str = "",
) -> list:
    """Recursively parse an h5 file and return a list of addresses for all datasets found

    Args:
        d: the hdf5 node from which to parse
        prefix: parent . Defaults to ''.

    Returns:
        _description_
    """
    key_list = []
    for k in d.keys():
        try:
            [key_list.append(s) for s in parse_h5_keys(d[k], prefix=prefix + "/" + k)]
        except AttributeError:
            key_list.append(prefix + "/" + k)
    return key_list


def get_h5_attrs(
    filename: str,
) -> dict:
    """Get the attributes from an h5 file"""
    with h5py.File(filename, "r", swmr=True) as h5file:
        attrs = {}
        inst_group = h5file["Entry/Instrument"]
        addresses = (a[1:] for a in parse_h5_keys(h5file["Entry/Instrument"]))
        for address in addresses:
            vals = inst_group[address][()]
            if "Info" in address:
                attrs = {
                    **attrs,
                    **{
                        f"Detector/{k}": reformat(v)
                        for k, v in [
                            tuple(s.split(":"))
                            for s in vals.decode("utf-8").split("\r\n")
                            if ":" in s
                        ]
                    },
                }
            elif "Other" in address:
                for v in vals:
                    name = v[0].decode("utf-8")
                    value = v[1].decode("utf-8")
                    type_ = v[3].decode("utf-8").lower()
                    if type_ == "integer":
                        value = int(value)
                    elif type_ == "double":
                        value = float(value)
                    elif type_ == "bool":
                        if value == "true":
                            value = True
                        else:
                            value = False
                        # value = True if value == 'true' else False
                    else:
                        raise ValueError(f"cannot interpret type {type_}")
                    attrs[f"Detector/{name}"] = value
            elif "Positioner" in address:
                pass
            else:
                attrs[address] = inst_group[address][()]
    return attrs
