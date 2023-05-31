from smartscan.virtualSGM4 import VirtualSGM4
import time

if __name__ == '__main__':
    vm = VirtualSGM4(
        'localhost', 
        12345, 
        ndim = 2, 
        limits=[0.0, 110.0, -100.0, 10.0], 
        filename = r"D:\data\SGM4\testing\controller_9.h5",
        verbose=True,
    )
    t0 = time.time()
    lin_pts = [
        (0,0),
        # (100,0),(200,0),(300,0),(400,0),(500,0),
        # (0,100),(100,100),(200,100),(300,100),(400,100),(500,100),
        # (0,200),(100,200),(200,200),(300,200),(400,200),(500,200),
        # (0,300),(100,300),(200,300),(300,300),(400,300),(500,300),
    ]

    vm.queue=lin_pts
    vm.wait_at_queue_empty = True

    vm.run()
    # print('All done. Quitting...')
