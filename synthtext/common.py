import sys
import signal
import random
import numpy as np
from contextlib import contextmanager


def set_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        pass  #raise TimeoutException

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# http://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
class TimeoutException(Exception):
    pass
