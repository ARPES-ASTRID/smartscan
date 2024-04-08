import logging
from pathlib import Path

import yaml

from smartscan.utils import ColoredFormatter
from smartscan.virtualSGM4 import VirtualSGM4

if __name__ == '__main__':

    with open('simulator_config.yaml', 'r') as f:
        settings = yaml.safe_load(f)

    file_name = settings['data']['file_name']
    data_dir = Path(settings['data']['data_dir'])
    if not data_dir.exists():
        data_dir = Path("F" + str(data_dir)[1:])
    assert data_dir.exists(), f'Data directory {data_dir} does not exist!'
    source_file = (data_dir / f'{file_name}').with_suffix('.h5')
    assert source_file.exists(), f'File {source_file} does not exist!'

    # init logger

    logger = logging.getLogger('virtualSGM4')
    logger.setLevel(settings['logging']['level'].upper())
    formatter = ColoredFormatter(settings['logging']['formatter'])

    sh = logging.StreamHandler()
    sh.setLevel(settings['logging']['level'].upper())
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info('Created logger.')
    vm = VirtualSGM4(
        'localhost', 
        54333, 
        logger=logger,
        simulate_times=settings['scan']['simulate_times'],
        save_to_file=settings['scan']['save_to_file'],
        dwell_time=settings['scan']['dwell_time'],
    )
    vm.init_scan_from_file(filename=source_file)
    logger.info(f'Initialized scan from file {source_file}')
    filedir = Path(settings['data']['target_dir'])
    i=0
    while True:
        filename = filedir / f'{file_name}_virtual_{i:03.0f}.h5'
        if not filename.exists():
            break
        else:
            i += 1
    
    vm.create_file(
        mode='x',
        filename=filename
    )

    # vm.current_pos = np.mean(vm.limits[:2]), np.mean(vm.limits[2:])
    # logger.info(f'Set current pos to center position: {vm.current_pos}')

    vm.run()
    logger.info('All done. Quitting...')
