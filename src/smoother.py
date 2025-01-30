import os
import numpy as np
import soundfile as sf
import scipy.fftpack as fft
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import chirp


def analyze_audio(file_path):
    data, samplerate = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    spectrum = np.abs(fft.fft(data))
    freqs = fft.fftfreq(len(data), 1/samplerate)
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    spectrum = spectrum[pos_mask]
    top_indices = np.argsort(spectrum)[-6:][::-1]
    top_freqs = freqs[top_indices]
    top_values = spectrum[top_indices]
    return samplerate, top_freqs, top_values

audio_dir = "../Audio/OOCC"
harmonics_data = []
for filename in sorted(os.listdir(audio_dir)):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_dir, filename)
        samplerate, top_freqs, top_values = analyze_audio(file_path)
        harmonics_data.append([filename] + list(top_freqs) + list(top_values))
        plt.figure()
        plt.plot(top_freqs, top_values, 'o', label=f'{filename} Harmonics')
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"Top Harmonics of {filename}")
        plt.legend()
        plt.show()
