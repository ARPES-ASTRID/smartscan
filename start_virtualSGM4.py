from smartscan.virtualSGM4 import VirtualSGM4
import time
import numpy as np
from pathlib import Path
import asyncio

if __name__ == '__main__':

    # source_file = r"D:\data\SGM4 - 2022 - CrSBr\data\Kiss05_15_1.h5"
    source_file =  Path(r"D:\data\SGM4\SmartScan\Z006_46.h5")
    source_file =  Path(r"D:\data\SGM4\SmartScan\Z006_35_0.h5")

    name = Path(source_file).stem

    vm = VirtualSGM4(
        'localhost', 
        12345, 
        verbose=True,
    )
    vm.init_scan_from_file(filename=source_file)
    filedir = Path(
        r"C:\Users\stein\OneDrive\Documents\_Work\_code\ARPES-ASTRID\smartscan\data"
        )
    i=0
    while True:
        filename = filedir / f'{name}_virtual_{i:03.0f}.h5'
        if not filename.exists():
            break
        else:
            i += 1
    
    vm.create_file(
        mode='x',
        filename=filename
    )

    vm.current_pos = np.mean(vm.limits[:2]), np.mean(vm.limits[2:])
    print('set current pos to', vm.current_pos)

    vm.run()
    print('All done. Quitting...')
