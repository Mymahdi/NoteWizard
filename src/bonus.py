from scipy.io.wavfile import read

fs, generated_signal = read('../Audio/noteHarryPotter.wav')

generated_signal = generated_signal / np.max(np.abs(generated_signal))

print("Audio file loaded successfully for analysis.")
