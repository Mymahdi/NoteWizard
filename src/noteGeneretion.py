import numpy as np
from scipy.io.wavfile import write


fs = 44100  
silence_duration = 0.025
final_signal = []

# Base frequencies for notes 
note_frequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
    'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
    'A#': 466.16, 'B': 493.88
}

# frequency for a given note and octave
def get_frequency(note, octave):
    base_freq = note_frequencies[note]
    return base_freq * (2 ** (octave - 4))

# Define Harry Potter notes
noteHarryPotter = [
    'B 4 0.3', 'E 5 0.6', 'G 5 0.2', 'F# 5 0.3', 'E 5 0.6',
    'B 5 0.4', 'A 5 0.8', 'F# 5 0.8', 'E 5 0.6', 'G 5 0.2',
    'F# 5 0.3', 'D# 5 0.7', 'F 5 0.4', 'B 4 1.6', 'B 4 0.3',
    'E 5 0.6', 'G 5 0.2', 'F# 5 0.3', 'E 5 0.6', 'B 5 0.4',
    'D 6 0.6', 'C# 6 0.3', 'C 6 0.6', 'G# 5 0.3', 'C 5 0.5',
    'B 5 0.2', 'A# 5 0.3', 'A# 4 0.6', 'G 5 0.3', 'E 5 1.6',
    'G 5 0.3', 'B 5 0.6', 'G 5 0.3', 'B 5 0.6', 'G 5 0.3',
    'C 6 0.6', 'B 5 0.3', 'A# 5 0.6', 'F# 5 0.3', 'G 5 0.5',
    'B 5 0.2', 'A# 5 0.3', 'A# 4 0.6', 'B 4 0.4', 'B 5 1.6',
    'G 5 0.3', 'B 5 0.7', 'G 5 0.3', 'B 5 0.7', 'G 5 0.3',
    'D 6 0.7', 'C# 6 0.3', 'C 6 0.8', 'G# 5 0.3', 'C 6 0.6',
    'B 5 0.2', 'A# 5 0.3', 'A# 4 0.6', 'G 5 0.4', 'E 5 1', 'E 5 1.6'
]

# Parse notes and calculate frequencies
melody = []
durations = []

for note_entry in noteHarryPotter:
    note, octave, duration = note_entry.split()
    melody.append(get_frequency(note, int(octave)))
    durations.append(float(duration))



for i, freq in enumerate(melody):
    # Generate time array for the note
    t = np.linspace(0, durations[i], int(fs * durations[i]), endpoint=False)
    
    # Generate sine wave for the note
    note_signal = np.sin(2 * np.pi * freq * t)
    
    # Append note signal to the final signal
    final_signal.extend(note_signal)
    
    # apply silence
    silence = np.zeros(int(fs * silence_duration))
    final_signal.extend(silence)

print("Generating sine waves for notes is DONE!!")

