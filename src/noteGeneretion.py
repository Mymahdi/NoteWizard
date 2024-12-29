import numpy as np
from scipy.io.wavfile import write

# Define the sampling rate
fs = 44100  # Sampling rate in Hz

# Frequencies of notes (for Octave 0)
frequencies = [
    16.352, 17.324, 18.354, 19.445, 20.602, 21.827, 23.125,
    24.500, 25.957, 27.500, 29.135, 30.868
]  # Frequencies of notes in Hz
print("Base structure set up for generating musical notes.")


# Example melody and durations
melody = [27.500, 30.868, 25.957, 24.500, 27.500]
durations = [0.5, 0.5, 0.5, 0.5, 0.5]
silence_duration = 0.025

print("Melody and durations defined.")

