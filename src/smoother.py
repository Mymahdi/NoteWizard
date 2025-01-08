import os
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt


directory = '../Audio/Octave5'
harmonics_data = {} 


for file in sorted(os.listdir(directory)):
    if file.endswith('.wav'):
        note_name = file.split('.')[0]

        fs, signal = read(os.path.join(directory, file))

        signal = signal / np.max(np.abs(signal))

        fft_spectrum = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / fs)
        magnitude = np.abs(fft_spectrum)

        # Limit to positive frequencies
        positive_freqs = frequencies[:len(frequencies) // 2]
        positive_magnitude = magnitude[:len(magnitude) // 2]

        # plt.figure(figsize=(10, 6))
        # plt.plot(positive_freqs, positive_magnitude, label=f'{note_name} Spectrum')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude')
        # plt.title(f'Fourier Transform of {note_name}')
        # plt.grid()
        # plt.legend()
        # plt.show()

        peaks_indices = positive_magnitude.argsort()[-7:][::-1]  # Top 7 including DC
        peaks_indices = [idx for idx in peaks_indices if positive_freqs[idx] > 0][:6]  # Exclude DC

        harmonics_data[note_name] = [(positive_freqs[idx], positive_magnitude[idx]) for idx in peaks_indices]
