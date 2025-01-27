import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

def find_closest_note(frequency):
    min_diff = float('inf')
    closest_note = None
    for note, base_freq in note_frequencies.items():
        for octave in range(1, 9):  # Search across octaves
            target_freq = base_freq * (2 ** (octave - 4))
            diff = abs(target_freq - frequency)
            if diff < min_diff:
                min_diff = diff
                closest_note = (note, octave)
    return closest_note

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

num_harmonics = 6 
peak_indices = np.argsort(positive_magnitude)[-num_harmonics:]
dominant_frequencies = [positive_freqs[idx] for idx in peak_indices]
dominant_frequencies.sort()
print("Dominant frequencies:", dominant_frequencies)


identified_notes = [find_closest_note(freq) for freq in dominant_frequencies]
print("Identified notes:", identified_notes)

original_notes = [(note.split()[0], int(note.split()[1])) for note in noteHarryPotter]

comparison = list(zip(original_notes, identified_notes))
for orig, ident in comparison:
    print(f"Original: {orig}, Identified: {ident}")

# Calculate accuracy of note identification
correct_matches = sum(1 for orig, ident in comparison if orig == ident)
accuracy = (correct_matches / len(original_notes)) * 100

print(f"Accuracy of note identification: {accuracy:.2f}%")

# Save results to a file
with open("note_identification_report.txt", "w") as report_file:
    report_file.write("Note Identification Report\n")
    report_file.write("=========================\n")
    for orig, ident in comparison:
        report_file.write(f"Original: {orig}, Identified: {ident}\n")
    report_file.write(f"\nAccuracy: {accuracy:.2f}%\n")

