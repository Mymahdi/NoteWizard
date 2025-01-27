import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

fs, generated_signal = read('../Audio/noteHarryPotter.wav')

generated_signal = generated_signal / np.max(np.abs(generated_signal))

print("Audio file loaded successfully for analysis.")

fft_spectrum = np.fft.fft(generated_signal)
frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / fs)
magnitude = np.abs(fft_spectrum)

positive_freqs = frequencies[:len(frequencies) // 2]
positive_magnitude = magnitude[:len(magnitude) // 2]

plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_magnitude, label="Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Fourier Transform of Generated Signal")
plt.grid()
plt.legend()
plt.show()

