import time
import datetime


class Timer:
    "Timer - class for timing things"

    def __init__(self):
        self.started = time.time()

    def reset(self):
        self.__init__()

    def elapsed(self):
        return datetime.timedelta(seconds=time.time() - self.started)

    def __str__(self):
        return str(self.elapsed())
