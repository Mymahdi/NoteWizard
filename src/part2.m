% Directory containing the audio files
audio_dir = '../Audio/Octave5';
notes = {'C5', 'C#D', 'D5', 'D#E', 'E5', 'F5', 'F#G', 'G5', 'G#A', 'A5', 'A#B', 'B5'};

% Initialize a structure to store coefficients
coefficients = struct();
for i = 1:length(notes)
    coefficients.(notes{i}) = [];
end

% Function to compute Fourier Transform and extract top 6 harmonics
function top_6_magnitudes = extract_harmonics(file_path)
    % Read the audio file
    [data, sample_rate] = audioread(file_path);
    
    % Compute the Fourier Transform
    N = length(data);
    yf = fft(data);
    xf = (0:N-1) * (sample_rate / N); % Frequency vector
    
    % Take only positive frequencies
    positive_freq = xf(1:floor(N/2));
    magnitudes = abs(yf(1:floor(N/2)));
    
    % Find the indices of the top 6 harmonics (excluding the DC component)
    [~, sorted_indices] = sort(magnitudes, 'descend');
    top_6_indices = sorted_indices(2:7); % Exclude the highest (DC component)
    top_6_magnitudes = magnitudes(top_6_indices);
end

% Process each note
for i = 1:length(notes)
    file_path = fullfile(audio_dir, [notes{i} '.wav']);
    if exist(file_path, 'file')
        top_6_magnitudes = extract_harmonics(file_path);
        coefficients.(notes{i}) = top_6_magnitudes;
    else
        fprintf('File %s not found.\n', file_path);
    end
end

% Display the coefficients
disp(coefficients);

% Convert the structure to a table for Excel export
note_names = fieldnames(coefficients);
max_length = max(cellfun(@(x) length(coefficients.(x)), note_names));
data = NaN(max_length, length(note_names)); % Initialize with NaN for missing values

for i = 1:length(note_names)
    values = coefficients.(note_names{i});
    data(1:length(values), i) = values;
end

% Create a table and export to Excel
T = array2table(data, 'VariableNames', note_names);
writetable(T, 'harmonics_coefficients.xlsx');
disp('Coefficients saved to harmonics_coefficients.xlsx');