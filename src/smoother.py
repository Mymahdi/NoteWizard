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

        # Perform Fourier Transform
        fft_spectrum = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / fs)
        magnitude = np.abs(fft_spectrum)

        # Limit to positive frequencies
        positive_freqs = frequencies[:len(frequencies) // 2]
        positive_magnitude = magnitude[:len(magnitude) // 2]

        # Plot the Fourier spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(positive_freqs, positive_magnitude, label=f'{note_name} Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Fourier Transform of {note_name}')
        plt.grid()
        plt.legend()
        plt.show()
