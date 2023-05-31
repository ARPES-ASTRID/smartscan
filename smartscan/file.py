from typing import Any, Tuple, Union, List, Sequence, Dict, Callable
import tqdm
import itertools
import numpy as np
import h5py
from tqdm.auto import trange
from . import processing

class SGM4FileReader:
    """Reads SMG4 files."""
    INVALID_POS = -999999999.0

    def __init__(self, filename: str, swmr: bool=True) -> None:
        """Initialize reader.

        Args:
            filename: path to file to read
        """
        self.filename = filename
        self.file = None
        self.swmr = swmr
        self._stack = None
        self._data = None
        self._positions = None

    def __enter__(self) -> Any:
        """Open file."""
        self.open()
        return self
    
    def __exit__(self, *args) -> None:
        """Close file."""
        self.close()

    def __del__(self) -> None:
        """Close file."""
        self.close()

    def open(self) -> None:
        """Open file."""
        self.file = h5py.File(self.filename, 'r', swmr=self.swmr)
    
    def close(self) -> None:
        """Close file."""
        if self.file is not None:
            self.file.close()

    def __len__(self) -> int:
        """Get number of spectra in file."""
        return self.file['Entry/Data/TransformedData'].shape[0]

    def get_data(self,index: int | slice) -> np.ndarray:
        """Get data from file.

        Args:
            index: index of spectrum to get. If slice, get slice of data
                if int, get single spectrum

        Returns:
            data: data from file
        """
        ds = self.file.get('Entry/Data/TransformedData')
        assert ds is not None, 'File does not contain data'
        # cache data
        if self._data is None:
            self._data = np.zeros(ds.shape)
        elif self._data.shape != ds.shape: 
            # resize if shape is different
            old = self._data
            self._data = np.zeros(ds.shape)
            self._data[:old.shape[0],:old.shape[1]] = old
        if isinstance(index, int):
            index = slice(index,None,None)
        # read data from file if not cached
        if self._data[index].sum() == 0:
            self._data[index] = ds[index]
        return self._data[index]
    
    def get_positions(self, index: int | slice) -> np.ndarray:
        """Get positions from file.

        Args:
            index: index of position to get. If slice, get slice of data

        Returns:
            positions: positions from file
        """
        ds = self.file.get('Entry/Data/ScanDetails/SetPositions')
        assert ds is not None, 'File does not contain positions'
        # cache positions
        if self._positions is None:
            self._positions = np.zeros(ds.shape)
        elif self._positions.shape != ds.shape: 
            # resize if shape is different
            old = self._positions
            self._positions = np.zeros(ds.shape)
            self._positions[:old.shape[0],:old.shape[1]] = old
        if isinstance(index, int):
            index = slice(index,None,None)
        # read positions from file if not cached
        if self._positions[index].sum() == 0:
            self._positions[index] = ds[index]
        return self._positions[index]

    def get_reduced_data(self, index: int, func:str | Callable='mean') -> dict:
        """Get reduced data from file.
        
        combine positions and reduced data to a list of 1D np.array
        
        Args:
            index: index of spectrum to get. If slice, get slice of data
                if int, get single spectrum
                
        Returns:
            data: data from file
        """
        data = self.get_data(index)
        positions = self.get_positions(index)
        reduced = [func(d) for d in tqdm(data,desc='Reducing spectra',leave=False)]
        out = {}
        # get the mean of data with the same position
        for p,r in zip(positions,reduced):
            if p in out:
                out[p].append(r)
            else:
                out[p] = [r]
        # combine data with the same position
        return {k:np.mean(v,axis=0) for k,v in out.items()}
    
    @property
    def stack(self) -> np.ndarray:
        """Get stack from file."""
        if self._stack is None:
            raise ValueError('Stack has not been loaded yet. Run reduce() first.')
        else:
            return self._stack

    def reduce(self, func: Union[callable, str]) -> np.ndarray:
        """Reduce each spectrum to a smaller feature space.

        Args:
            func: function to reduce stack with

        Returns:
            reduced: reduced stack
        """
        if hasattr(np, func): # check if function is a numpy function
            func = getattr(np, func)
        elif callable(func):
            pass
        elif hasattr(processing, func): # check if function is a function in the reduce module
            func = getattr(processing, func)
        else:
            raise ValueError("Function is not a numpy function nor a callable nor a "/
                             "function in the reduce module")
        for i in trange(len(self), desc='Reducing spectra'):
            reduced = func(self.get_data(i))
            if not isinstance(reduced, np.ndarray):
                reduced = np.array(reduced)
            if reduced.ndim == 0:
                reduced = reduced.reshape((1,))
            elif reduced.ndim >= 2:
                reduced = reduced.flatten()
            # cache reduced stack
            if self._stack is None:
                self._stack = np.empty((len(self),len(reduced)+self.ndim))
            if self._stack.shape[1] != len(reduced)+self.ndim:  
                # resize if shape is different
                old = self._stack
                self._stack = np.empty((len(self),len(reduced)+self.ndim))
                self._stack[:old.shape[0],:old.shape[1]] = old

            pos = self.get_positions(i)
            if pos.ndim == 0:
                pos = pos.reshape((1,))
            elif pos.ndim >= 2:
                pos = pos.flatten()
    
            self._stack[i,...] = np.concatenate([pos,reduced])

        return self._stack
    
    @property
    def raw_data(self) -> np.ndarray:
        """Get stack from file."""
        return self.file["Entry/Data/TransformedData"][()]
    
    @property
    def raw_dims(self) -> Tuple[int]:
        """Get shape of spectra."""
        dims = self.file['Entry/Data/ScanDetails/SlowAxis_names'][()]
        return dims

    @property
    def raw_shape(self) -> Tuple[int]:
        return self.file['Entry/Data/ScanDetails/SlowAxis_names'][()]

    @property
    def positions(self) -> Tuple[np.ndarray]:
        """Get positions from file.

        Also substitutes invalid positions with the previous valid position.

        Returns:
            positions: array of positions
        """
        try:
            pos_array = self.file['Entry/Data/ScanDetails/SetPositions'][()]
            corrected = []
            previous = None
            for line in pos_array:
                assert len(line) == self.ndim, 'length of position does not match dimensionality'
                corr_line = []
                for i, p in enumerate(line):
                    if p == self.INVALID_POS:
                        if previous is None:
                            raise ValueError('First position is invalid')
                        else:
                            p = previous[i]
                    corr_line.append(p)
                corrected.append(corr_line)
                previous = corr_line
        except KeyError:
            Warning('File does not contain positions. Probably loading old data. Using axes instead.')
            corrected =  itertools.product(*self.axes)[:len(self)]
        return np.array(corrected)
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions from file."""
        ndim = int(self.file['Entry/Data/ScanDetails/Dimensions'][()])
        return ndim
    
    @property
    def limits(self) -> List[Tuple[float]]:
        """Get limits from file."""
        lengths = self.file['Entry/Data/ScanDetails/SlowAxis_length'][()]
        starts = self.file['Entry/Data/ScanDetails/SlowAxis_start'][()]
        steps = self.file['Entry/Data/ScanDetails/SlowAxis_step'][()]
        assert len(lengths) == len(starts) == len(steps) == self.ndim, \
            'lengths of limits and dimensionality do not match'
        limits = [(start, start + step * (length - 1)) for start, step, length in zip(starts, steps, lengths)]
        return limits
    
    @property
    def map_shape(self):
        """Get shape of map."""
        lengths = self.file['Entry/Data/ScanDetails/SlowAxis_length'][()]
        return tuple(lengths)
    
    @property
    def axes(self) -> List[np.ndarray]:
        """Get coordinate axes from file."""
        lengths = self.file['Entry/Data/ScanDetails/SlowAxis_length'][()]
        starts = self.file['Entry/Data/ScanDetails/SlowAxis_start'][()]
        steps = self.file['Entry/Data/ScanDetails/SlowAxis_step'][()]
        axes = [np.linspace(start, start + step * (length -1), length) for start, step, length in zip(starts, steps, lengths)]
        return axes
    
    @property
    def coords(self) -> Dict[str, np.ndarray]:
        """Get coordinates from file."""
        coords = {dim:axis for dim, axis in zip(self.dims, self.axes)}
        return coords
    
    @property
    def dims(self) -> np.ndarray:
        """Get dimensions from file."""
        dims = self.file['Entry/Data/ScanDetails/SlowAxis_names'][()]
        dims = [d.decode('utf-8') for d in dims]
        return dims
    
    @property
    def dwell_time(self) -> float:
        """Get dwell time from file."""
        return 1.1 # TODO: alfred will fix
    
    def to_xarray(self) -> np.ndarray:
        """Unravel stack into an nD array."""
        raise NotImplementedError
    
    def nearest(self, position: Sequence[float]) -> Tuple[int]:
        """Find nearest position in the grid.

        Args:
            position: position to find
        
        Returns:
            index: index of nearest position
        """
        assert len(position) == self.ndim, 'length of position does not match dimensionality'
        index = []
        for pos, axis in zip(position, self.axes):
            assert pos >= axis[0] and pos <= axis[-1], 'position is outside of limits'
            index.append(np.argmin(np.abs(axis - pos)))
        return tuple(index)
    

if __name__ == "__main__":
    test_data = "D:\data\SGM4 - example\Testing\Controller_9.h5"
    with SGM4FileReader(test_data) as reader:
        print(f'spectra shape {reader.spectra.shape}\n')
        print(f'ndim {reader.ndim}\n')
        print(f'limits {reader.limits}\n')
        print(f'axes {reader.axes}\n')
        print(f'dims {reader.dims}\n')
        print(f'coords {reader.coords}\n')
        print(f'spectra_dims {reader.spectra_dims}\n')
        print(f'spectra_shape {reader.spectra_shape}\n')
        print(f'positions {reader.positions.shape}\n')
        print(f'positions {reader.positions[:,:10]}\n')

