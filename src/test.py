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

def calculate_similarity(file1_props, file2_props):
    # Define weights for each property (adjust as needed)
    weights = {
        "channels": 0.2,
        "sample_width": 0.2,
        "frame_rate": 0.3,
        "frame_count": 0.2,
        "duration": 0.1,
    }

    similarity_score = 0

    # Compare each property and calculate similarity
    for key in file1_props:
        if key in file2_props:
            if file1_props[key] == file2_props[key]:
                similarity_score += weights[key]
            else:
                # If the property is numeric, calculate a partial match
                if isinstance(file1_props[key], (int, float)):
                    max_value = max(file1_props[key], file2_props[key])
                    min_value = min(file1_props[key], file2_props[key])
                    if max_value != 0:
                        similarity_score += weights[key] * (min_value / max_value)
                else:
                    # For non-numeric properties, no partial match
                    pass

    # Convert the score to a percentage
    similarity_percentage = similarity_score * 100
    return similarity_percentage

# Get properties of the two files
file1_props = get_wav_properties("1.wav")
file2_props = get_wav_properties("noteHarryPotter.wav")

# Calculate similarity percentage
similarity_percentage = calculate_similarity(file1_props, file2_props)

# Print the result
print(f"Similarity between the two files: {similarity_percentage:.2f}%")

# Print detailed properties
print("\nFile 1 properties:")
for key, value in file1_props.items():
    print(f"{key}: {value}")

print("\nFile 2 properties:")
for key, value in file2_props.items():
    print(f"{key}: {value}")