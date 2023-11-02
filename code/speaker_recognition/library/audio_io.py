from dataclasses import dataclass

# Add-on packages
import numpy as np
import wave

@dataclass
class AudioData:
    """
    AudioData - contains
       samples - N x channels array
       Fs - sample rate (Hz)
    """
    samples: np.array
    Fs: int

# -------------------- file I/O -----------------------------------

def read_wav(filename):
    """
    Read audio data from wav file
    :param filename:  file to read
    :return: AudioData instance
    """
    # Open a audio file reader.
    # Note the sample reader refers to each sample as a frame.
    # This is not uncommon, but is different from the frames discussed
    # in class which are groups of samples.
    reader = wave.open(filename, 'rb')
    # Extract sample rate
    Fs = reader.getframerate()
    # Extract sample properties: channel count, sample count,
    # and number of bits used to represent the sample
    channels_N = reader.getnchannels()
    samples_N = reader.getnframes()
    sample_width_bytes = reader.getsampwidth()
    bits_per_byte = 8
    sample_width_bits = reader.getsampwidth() * bits_per_byte

    # Get byte array representing audio
    buffer = reader.readframes(samples_N)

    # Map of bit widths to numpy library types
    # Note:  No support for 24 bit signals which do not align on word boundaries
    bitwidths = {
        8: np.int8,
        16: np.int16,
        32: np.int32
    }
    if sample_width_bits not in bitwidths:
        raise ValueError(f"Bit widths of {sample_width_bits} not supported")

    # Convert to 1D array
    raw_samples = np.frombuffer(buffer, dtype=bitwidths[sample_width_bits])
    # Reshape by number of channels
    samples = np.reshape(raw_samples, (samples_N, channels_N), order='C')

    return AudioData(samples, Fs)

