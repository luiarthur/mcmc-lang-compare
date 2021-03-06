import time

class Timer(object):
    """
    Usage:
    with Timer('Model training'):
        time.sleep(2)
        x = 1
    """
    def __init__(self, name=None, digits=0):
        self.name = name
        self.digits = digits

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(self.name, end=' ')

        elapsed = time.time() - self.tstart
        print('time: {}s'.format(round(elapsed, self.digits)))

