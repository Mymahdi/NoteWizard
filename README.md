# 1. Generating Harry Potter Theme Notes in Python

## Introduction
This Python script reads musical notes from a MATLAB file (`notes.m`), converts them into corresponding frequencies, generates a waveform for each note, and saves the resulting sound as a `.wav` file.

---

## Dependencies
```python
import numpy as np
from scipy.io.wavfile import write
import re
```
- `numpy` for numerical computations and signal generation.
- `scipy.io.wavfile.write` for saving the generated waveform as an audio file.
- `re` for extracting note data from `notes.m`.

---

## Sampling Configuration
```python
fs = 44100  
silence_duration = 0.025
final_signal = []
noteHarryPotter = []
```
- The sampling rate is set to 44.1 kHz.
- A short silence (0.025s) is inserted between notes to enhance clarity.

---

## Defining Note Frequencies
```python
note_frequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
    'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
    'A#': 466.16, 'B': 493.88
}
```
- This dictionary stores the standard frequencies for each note in the 4th octave.

---

## Function to Calculate Frequency
```python
def get_frequency(note, octave):
    base_freq = note_frequencies[note]
    return base_freq * (2 ** (octave - 4))
```
- Converts a note into its corresponding frequency based on octave number.

---

## Reading Notes from `notes.m`
```python
with open('notes.m', 'r') as file:
    content = file.read()
    matches = re.findall(r"'(.*?)'", content)
    noteHarryPotter.extend(matches)
```
- Extracts notes from `notes.m` using regex.
- Notes are stored in `noteHarryPotter` list.

---

## Processing Notes and Durations
```python
melody = []
durations = []
for note_entry in noteHarryPotter:
    note, octave, duration = note_entry.split()
    melody.append(get_frequency(note, int(octave)))
    durations.append(float(duration))
```
- Splits each entry into note name, octave, and duration.
- Converts the note into its corresponding frequency.

---

## Generating Waveform
```python
for i, freq in enumerate(melody):
    t = np.linspace(0, durations[i], int(fs * durations[i]), endpoint=False)
    note_signal = np.sin(2 * np.pi * freq * t)
    final_signal.extend(note_signal)
    silence = np.zeros(int(fs * silence_duration))
    final_signal.extend(silence)
```
- Creates sine waves for each note based on its duration.
- Appends a brief silence between notes for clarity.

---

## Normalization and Saving
```python
final_signal = np.array(final_signal)
final_signal = final_signal / np.max(np.abs(final_signal))
write('../Audio/noteHarryPotter.wav', fs, (final_signal * 32767).astype(np.int16))
print("Audio file saved successfully.")
```
- Converts the list into a NumPy array.
- Normalizes the amplitude to avoid clipping.
- Saves the generated waveform as `noteHarryPotter.wav`.

---

## Output
- The script generates `noteHarryPotter.wav` in the `../Audio/` directory.
- This file contains the synthesized melody based on the `notes.m` file.

---

## Usage
Run the script:
```sh
python script.py
```
Ensure `notes.m` is present in the same directory and contains properly formatted note data.

