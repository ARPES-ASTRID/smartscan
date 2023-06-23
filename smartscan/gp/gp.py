from typing import Callable, Sequence, Tuple, Union
from tqdm.auto import tqdm
import numpy as np
import time

from ..utils import manhattan_distance, closest_point_on_int_grid

def measurement_loop(
    init_positions: Sequence[Sequence[float]], 
    init_values: Sequence[Sequence[float]],
    gp,
    measure_func: Callable,
    train_pars: dict = None,
    ask_pars: dict = None,
    n_points: int = 200,
    train_every: int = 20,
    train_at: Sequence[int]= [20,40,80,160,320,640,1280,2560,5120,10240],
    verbose: bool=False,
    return_times: bool=False,
) -> None:
    if ask_pars is None:
        ask_pars = {}
    if train_pars is None:
        train_pars = {}
    
    values = init_values.copy()
    positions = init_positions.copy()
    times = []
    for i in tqdm(range(n_points), desc="Acquisition loop"):
        gp.tell(positions, values)
        print(f'positions: {positions}')
        print(f'values: {values}')
        next_pt = gp.ask(position=positions[-1],**ask_pars)
        next_on_grid = closest_point_on_int_grid(next_pt['x'])
        print(f'next point added {next_on_grid}: {next_pt["y"]}')
        next_pt.update({'pos':next_on_grid})
        next_val = measure_func(next_pt['pos'])
        # np.array(
        #     reduce(xdata[next_pt['pos'][0],next_pt['pos'][1]].values)
        # )[None,:]
        next_pos = np.array(next_pt['pos'])[None,:]
        if verbose:
            print(f'next point added {next_pos}: {next_val}')
        positions = np.append(positions, next_pos, axis=0)
        values = np.append(values, next_val, axis=0)
        if verbose:
            print(f'n points: {len(positions)}')
        if train_at is None:
            train_at = []
        if i % train_every == 0 or i in train_at:
            t0 = time.time()
            gp.train_gp(**train_pars)
            dt = time.time()-t0
            times.append(dt)
            if verbose:
                print(f"Training at i={i}. Took {dt:.2f} s")
    if return_times:
        return positions, values, times
    return positions, values
