import wave

def get_wav_properties(filename):
    with wave.open(filename, 'r') as wav_file:
        return {
            "channels": wav_file.getnchannels(),
            "sample_width": wav_file.getsampwidth(),
            "frame_rate": wav_file.getframerate(),
            "frame_count": wav_file.getnframes(),
            "duration": wav_file.getnframes() / wav_file.getframerate(),
        }

file1_props = get_wav_properties("1.wav")
file2_props = get_wav_properties("noteHarryPotter.wav")

if file1_props == file2_props:
    print("File properties match!")
else:
    print("File properties are different!")
    print(f"File 1 properties: {file1_props}")
    print(f"File 2 properties: {file2_props}")
