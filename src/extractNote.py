import numpy as np
from scipy.io.wavfile import read
import scipy.signal as signal

note_frequencies = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
}

def correlate_segment_with_note(segment, fs, note_freq):
    t = np.linspace(0, len(segment) / fs, len(segment), endpoint=False)
    template = np.sin(2 * np.pi * note_freq * t)
    segment_norm = segment - np.mean(segment)
    template_norm = template - np.mean(template)
    numerator = np.sum(segment_norm * template_norm)
    denominator = np.sqrt(np.sum(segment_norm**2) * np.sum(template_norm**2))
    if denominator == 0:
        return 0
    return numerator / denominator

fs, audio = read('../Audio/noteHarryPotter.wav')

if audio.ndim > 1:
    audio = audio[:, 0]

audio = audio / np.max(np.abs(audio))

threshold = 0.05
indices = np.where(np.abs(audio) > threshold)[0]

if indices.size == 0:
    print("No note segments found.")
    exit()

segments = []
start = indices[0]
prev = indices[0]
for idx in indices[1:]:
    if idx - prev > int(0.01 * fs):
        segments.append((start, prev))
        start = idx
    prev = idx
segments.append((start, prev))

extracted_notes = []

for (start, end) in segments:
    segment = audio[start:end]
    best_corr = -2
    best_note = None

    for note, freq in note_frequencies.items():
        corr = correlate_segment_with_note(segment, fs, freq)
        if corr > best_corr:
            best_corr = corr
            best_note = note

    if best_note is not None:
        extracted_notes.append(best_note)

print("Extracted Notes Array (using correlation):")
print(extracted_notes)