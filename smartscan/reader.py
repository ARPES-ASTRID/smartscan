from typing import Any, Tuple, Union, List, Sequence, Dict
import numpy as np
import h5py

class SGM4Reader:
    """Reads SMG4 files."""
    INVALID_POS = -999999999.0

    def __init__(self, filename: str) -> None:
        """Initialize reader.

        Args:
            filename: path to file to read
        """
        self.filename = filename
        self.file = None
        self.ndim = None
        self.limits = None
        self.current_pos = None
        
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
        self.file = h5py.File(self.filename, 'r', swmr=True)
    
    def close(self) -> None:
        """Close file."""
        if self.file is not None:
            self.file.close()

    @property
    def spectra(self) -> np.ndarray:
        """Get stack from file."""
        return self.file["Entry/Data/TransformedData"][()]
    
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
            raise KeyError('File does not contain positions. Probably loading old data.')
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
        limits = [(start, start + step * length) for start, step, length in zip(starts, steps, lengths)]
        return limits
    
    @property
    def axes(self) -> List[np.ndarray]:
        """Get coordinate axes from file."""
        lengths = self.file['Entry/Data/ScanDetails/SlowAxis_length'][()]
        starts = self.file['Entry/Data/ScanDetails/SlowAxis_start'][()]
        steps = self.file['Entry/Data/ScanDetails/SlowAxis_step'][()]
        axes = [np.linspace(start, start + step * length, length) for start, step, length in zip(starts, steps, lengths)]
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
    def spectra_dims(self) -> Tuple[int]:
        """Get shape of spectra."""
        dims = self.file['Entry/Data/ScanDetails/SlowAxis_names'][()]
        return dims

    @property
    def spectra_shape(self) -> Tuple[int]:
        return self.file['Entry/Data/ScanDetails/SlowAxis_names'][()]

    def to_xarray(self) -> np.ndarray:
        """Unravel stack into an nD array."""
        raise NotImplementedError
    
if __name__ == "__main__":
    test_data = "D:\data\SGM4 - example\Testing\Controller_9.h5"
    with SGM4Reader(test_data) as reader:
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
