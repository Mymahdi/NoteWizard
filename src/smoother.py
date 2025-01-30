import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import find_peaks

directory = '../Audio/Octave5'
plots_dir = '../Plots'
os.makedirs(plots_dir, exist_ok=True)

harmonic_data = []

for file in sorted(os.listdir(directory)):
    if file.endswith('.wav'):
        note_name = file.split('.')[0]

        fs, signal = read(os.path.join(directory, file))

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        signal = signal / np.max(np.abs(signal))

        fft_spectrum = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / fs)
        magnitude = np.abs(fft_spectrum)

        positive_freqs = frequencies[:len(frequencies) // 2]
        positive_magnitude = magnitude[:len(magnitude) // 2]

        positive_magnitude = np.nan_to_num(positive_magnitude, nan=0.0, posinf=0.0, neginf=0.0)

        peaks, _ = find_peaks(positive_magnitude, height=0.05 * np.max(positive_magnitude))

        peak_frequencies = positive_freqs[peaks]
        peak_magnitudes = positive_magnitude[peaks]
        
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        top_harmonics = sorted_indices[:6]
        
        harmonic_info = [note_name]
        for idx in top_harmonics:
            harmonic_info.append(peak_frequencies[idx])
            harmonic_info.append(peak_magnitudes[idx])
        
        harmonic_data.append(harmonic_info)
        
        plt.figure(figsize=(10, 6))
        plt.plot(positive_freqs, positive_magnitude, label=f'{note_name} Spectrum')
        plt.scatter(peak_frequencies[top_harmonics], peak_magnitudes[top_harmonics], color='red', marker='o', label='Harmonics')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'Fourier Transform of {note_name}')
        plt.grid()
        plt.legend()
        
        plot_filename = os.path.join(plots_dir, f'{note_name}_spectrum.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved: {plot_filename}")

columns = ['Note'] + [f'Harmonic {i+1} Freq (Hz)' for i in range(6)] + [f'Harmonic {i+1} Mag' for i in range(6)]
df = pd.DataFrame(harmonic_data, columns=columns)
excel_filename = '../Harmonics.xlsx'
df.to_excel(excel_filename, index=False)
print(f"Harmonic coefficients saved to {excel_filename}")