% Define the folder paths
baseFolder = 'Dataset/Training'; % Assuming base folder is the current working directory
testingBaseFolder = 'Dataset/Testing'; % Assuming base folder is the current working directory
regions = {'Hebron', 'Jerusalem', 'Nablus', 'Ramallah_Reef'};

% Initialize cell arrays to store the features and labels
allFeatures = {};
allLabels = {};

% Variable to track the maximum number of frames
maxNumFrames = 0;

% First pass to determine the maximum number of frames
for r = 1:length(regions)
    regionFolder = fullfile(baseFolder, regions{r});
    wavFiles = dir(fullfile(regionFolder, '*.wav'));
    
    fprintf('Processing folder: %s\n', regions{r});
    
    for k = 1:length(wavFiles)
        % Load the audio file
        filePath = fullfile(regionFolder, wavFiles(k).name);
        [audio, fs] = audioread(filePath);
        
        % Extract MFCC features
        features = extractMFCCFeatures(audio, fs);
        
        % Update the maximum number of frames
        maxNumFrames = max(maxNumFrames, size(features, 2));
        
        % Store the features and the label (region)
        allFeatures{end+1} = features;
        allLabels{end+1} = regions{r};
        
        fprintf('Processed file: %s\n', wavFiles(k).name);
    end
    
    fprintf('Finished processing folder: %s\n', regions{r});
end

% Determine the maximum number of frames
maxNumFrames = max(cellfun(@(x) size(x, 2), allFeatures));

% Second pass to pad or trim features and concatenate them
featuresMatrix = zeros(size(allFeatures{1}, 1), maxNumFrames, length(allFeatures));
for i = 1:length(allFeatures)
    numFrames = size(allFeatures{i}, 2);
    if numFrames < maxNumFrames
        % Pad features with zeros
        padding = zeros(size(allFeatures{i}, 1), maxNumFrames - numFrames);
        allFeatures{i} = [allFeatures{i}, padding];
    elseif numFrames > maxNumFrames
        % Trim features
        allFeatures{i} = allFeatures{i}(:, 1:maxNumFrames);
    end
    featuresMatrix(:,:,i) = allFeatures{i};
end

% Convert labels to categorical
labels = categorical(allLabels);

% Save features and labels to a .mat file
save('features_labels.mat', 'featuresMatrix', 'labels');

% Display the size of the feature matrix and the labels
disp(size(featuresMatrix));
disp(size(labels));

% Train GMM for each region
numComponents = 64;
gmmModels = cell(length(regions), 1);

for r = 1:length(regions)
    fprintf('Training GMM for region: %s\n', regions{r});
    
    % Extract features for the current region
    regionFeatures = featuresMatrix(:,:,strcmp(labels, regions{r}));
    
    % Convert the features to a 2D matrix where each row is a feature vector
    regionFeatures2D = reshape(regionFeatures, size(regionFeatures, 1), [])';
    
    % Check if there are enough data points for GMM training
    if size(regionFeatures2D, 1) < numComponents
        fprintf('Error: Insufficient data for region %s\n', regions{r});
        continue; % Skip training for this region
    end
    
    % Train GMM with fitgmdist function
    gmmModels{r} = fitgmdist(regionFeatures2D, numComponents,'CovarianceType', 'diagonal', 'RegularizationValue', 0.01, 'Replicates', 5);
    
    fprintf('Finished training GMM for region: %s\n', regions{r});
end

% Save the trained GMM models
save('gmmModels.mat', 'gmmModels', 'regions');

fprintf('GMM training completed and models saved.\n');

% Testing the trained GMM models
predictedLabels = {};
trueLabels = {};

% Loop over each region in the testing dataset
for r = 1:length(regions)
    regionFolder = fullfile(testingBaseFolder, regions{r});
    wavFiles = dir(fullfile(regionFolder, '*.wav'));
    
    fprintf('Testing region: %s\n', regions{r});
    
    % Iterate through each audio file in the region folder
    for k = 1:length(wavFiles)
        % Load the audio file
        filePath = fullfile(regionFolder, wavFiles(k).name);
        [audio, fs] = audioread(filePath);
        
        % Extract MFCC features
        features = extractMFCCFeatures(audio, fs);
        
        % Calculate log likelihood for each GMM model
        logLikelihoods = zeros(length(gmmModels), 1);
        for g = 1:length(gmmModels)
            % Calculate log likelihood
            logLikelihoods(g) = sum(log(pdf(gmmModels{g}, features)));
        end
        
        % Find the region with the maximum log likelihood
        [~, predictedRegionIdx] = max(logLikelihoods);
        
       % Store predicted and true labels
        predictedLabels{end+1} = regions{predictedRegionIdx};
        trueLabels{end+1} = regions{r};
        
        fprintf('Predicted region for file %s: %s\n', wavFiles(k).name, regions{predictedRegionIdx});
    end
end

% Calculate accuracy
correctPredictions = sum(strcmp(predictedLabels, trueLabels));
totalFiles = length(predictedLabels);
accuracy = correctPredictions / totalFiles * 100;

fprintf('Accuracy: %.2f%%\n', accuracy);

function features = extractMFCCFeatures(audio, fs)
    % Pre-emphasis filter
    pre_emphasis = 0.97;
    emphasized_audio = filter([1 -pre_emphasis], 1, audio);

    % Framing
    frame_size = 0.025; % 25 ms
    frame_stride = 0.01; % 10 ms
    frame_length = round(frame_size * fs);
    frame_step = round(frame_stride * fs);
    signal_length = length(emphasized_audio);
    num_frames = floor((signal_length - frame_length) / frame_step) + 1;

    % Zero padding
    pad_signal_length = num_frames * frame_step + frame_length;
    z = zeros(pad_signal_length - signal_length, 1);
    padded_signal = [emphasized_audio; z];

    % Windowing
    indices = repmat(1:frame_length, num_frames, 1) + repmat((0:num_frames-1)' * frame_step, 1, frame_length);
    frames = padded_signal(indices);
    hamming_window = hamming(frame_length);
    frames = frames .* hamming_window';

    % FFT and Power Spectrum
    NFFT = 512;
    mag_frames = abs(fft(frames, NFFT, 2));
    pow_frames = (1.0 / NFFT) * (mag_frames .^ 2);

    % Mel Filter Bank
    nfilt = 24;
    low_freq_mel = 0;
    high_freq_mel = (2595 * log10(1 + (fs / 2) / 700)); % Convert Hz to Mel
    mel_points = linspace(low_freq_mel, high_freq_mel, nfilt + 2); % Equally spaced in Mel scale
    hz_points = 700 * (10 .^ (mel_points / 2595) - 1); % Convert Mel to Hz
    bin = floor((NFFT + 1) * hz_points / fs);

    fbank = zeros(nfilt, NFFT / 2 + 1);
    for m = 2:nfilt+1
        f_m_minus = bin(m - 1);   % left
        f_m = bin(m);             % center
        f_m_plus = bin(m + 1);    % right

        for k = f_m_minus:f_m
            fbank(m-1, k+1) = (k - f_m_minus) / (f_m - f_m_minus);
        end
        for k = f_m:f_m_plus
            fbank(m-1, k+1) = (f_m_plus - k) / (f_m_plus - f_m);
        end
    end

    filter_banks = log((pow_frames(:, 1:NFFT/2+1) * fbank' + eps).^2);

    % DCT
    num_ceps = 12;
    mfcc = dct(filter_banks, [], 2);
    mfcc = mfcc(:, 1:num_ceps);

    % Energy of windowed signal
    energy = sum(frames .^ 2, 2);
    energy = log(energy + eps); % Log energy

    % Append energy to MFCCs
    mfcc = [mfcc energy];

    % Calculate delta
    mfcc_padded = [mfcc(1, :); mfcc; mfcc(end, :)]; % Pad the mfcc matrix
    delta = diff(mfcc_padded, 1, 1); % Compute first-order difference
    delta = delta(1:end-1, :); % Trim to match original size

    % Calculate delta-delta
    delta_padded = [delta(1, :); delta; delta(end, :)]; % Pad the delta matrix
    delta_delta = diff(delta_padded, 1, 1); % Compute first-order difference again
    delta_delta = delta_delta(1:end-1, :); % Trim to match original size

    % Transpose to make sure each feature type is aligned correctly
    mfcc = mfcc';
    delta = delta';
    delta_delta = delta_delta';

    % Combine MFCC, Delta, and Delta-Delta by concatenating
    features = [mfcc; delta; delta_delta];
end

