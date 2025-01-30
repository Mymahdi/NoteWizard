
import numpy as np
from scipy.io.wavfile import read
import os

def extract_initial_note(file_path):
    fs, signal = read(file_path)
    signal = signal / np.max(np.abs(signal))
    fft_spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_spectrum), 1 / fs)
    magnitude = np.abs(fft_spectrum)
    positive_freqs = frequencies[:len(frequencies) // 2]
    positive_magnitude = magnitude[:len(magnitude) // 2]
    peak_index = np.argmax(positive_magnitude)
    detected_freq = positive_freqs[peak_index]
    note_frequencies = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
        'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
        'A#': 466.16, 'B': 493.88
    }


    closest_note = min(note_frequencies, key=lambda note: abs(note_frequencies[note] - detected_freq))
    return detected_freq, closest_note

file_path = "../Audio/noteHarryPotter.wav"

if os.path.exists(file_path):
    detected_freq, closest_note = extract_initial_note(file_path)
    print(f"Detected Frequency: {detected_freq:.2f} Hz")
    print(f"Closest Note: {closest_note}")

else:
    print("File not found. Please check the path.")