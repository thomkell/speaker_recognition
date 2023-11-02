import math
from typing import Union

ms_per_s = 1000  # Number of milliseconds per second

def ms_to_samples(Fs: Union[float, int], duration_ms: round) -> int:
    """
    Convert milliseconds to samples
    :param Fs: sample rate
    :param duration_ms: duration in milliseconds
    :return: integer number of samples
    """

    return int(round(Fs * duration_ms / ms_per_s))
