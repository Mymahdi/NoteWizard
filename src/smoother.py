import os
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

# Directory containing Octave 5 note files
directory = '../Audio/Octave5'

# Iterate over all `.wav` files in the directory
for file in sorted(os.listdir(directory)):
    if file.endswith('.wav'):
        note_name = file.split('.')[0]

        # Read the audio file
        fs, signal = read(os.path.join(directory, file))

        # Normalize the signal
        signal = signal / np.max(np.abs(signal))

