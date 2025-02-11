import numpy as np
from scipy.io.wavfile import write
import re

fs = 44100  
silence_duration = 0.025
final_signal = []
noteHarryPotter = []

note_frequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
    'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
    'A#': 466.16, 'B': 493.88
}

def get_frequency(note, octave):
    base_freq = note_frequencies[note]
    return base_freq * (2 ** (octave - 4))

with open('notes.m', 'r') as file:
    content = file.read()
    matches = re.findall(r"'(.*?)'", content)
    noteHarryPotter.extend(matches)

melody = []
durations = []

for note_entry in noteHarryPotter:
    note, octave, duration = note_entry.split()
    melody.append(get_frequency(note, int(octave)))
    durations.append(float(duration))



for i, freq in enumerate(melody):
    t = np.linspace(0, durations[i], int(fs * durations[i]), endpoint=False)
    note_signal = np.sin(2 * np.pi * freq * t)
    final_signal.extend(note_signal)
    
    silence = np.zeros(int(fs * silence_duration))
    final_signal.extend(silence)

final_signal = np.array(final_signal)

final_signal = final_signal / np.max(np.abs(final_signal))

write('../Audio/noteHarryPotter.wav', fs, (final_signal * 32767).astype(np.int16))
print("Audio file saved successfully.")

