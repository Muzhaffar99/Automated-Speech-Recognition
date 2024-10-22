clear all;

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Commands Data 

% define the folder for the speech dataset in .wav format, 1 second files
datasetFolder = 'speechDataReduced';

% Create an audioDatastore that points to the data set
ads = audioDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');

%% Choose Words to Recognize

% define a subset of command words to recognise from the full data set
commands = categorical(["yes","no","up","down","left","right","on","off","stop","go"]);

% define an index into command words and unknown words
isCommand = ismember(ads.Labels,commands);
isUnknown = ~ismember(ads.Labels,[commands,"_background_noise_"]);

% specify the fraction of unknown words to include - labeling words that
% are not commands as |unknown| creates a group of words that approximates
% the distribution of all words other than the commands. The network uses
% this group to learn the difference between commands and all other words.
includeFraction = 0.1;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

% create a new data store of only command words and unknown words 
adsSubset = subset(ads,isCommand|isUnknown);

% count the number of instances of each word
countEachLabel(adsSubset)

%% Define Training, Validation, and Test data Sets

% define split proportion for training, validation and test data
p1 = 0.6; % training data proportion
p2 = 0.4; % validation data proportion
[adsTrain,adsValidation] = splitEachLabel(adsSubset,p1);

% reduce the dataset to speed up training 
numUniqueLabels = numel(unique(adsTrain.Labels));
nReduce = 4; % Reduce the dataset by a factor of nReduce
adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels / nReduce));
adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files) / numUniqueLabels / nReduce));

%% define object for computing auditory spectrograms from audio data

% spectrogram parameters
fs = 16e3;             % sample rate of the data set
segmentDuration = 1;   % duration of each speech clip (in seconds)
frameDuration = 0.025; % duration of each frame for spectrum calculation
hopDuration = 0.010;   % time step between each spectrum

segmentSamples = round(segmentDuration*fs); % number of segment samples
frameSamples = round(frameDuration*fs);     % number of frame samples
hopSamples = round(hopDuration*fs);         % number of hop samples
overlapSamples = frameSamples - hopSamples; % number of overlap samples

FFTLength = 512;  % number of points in the FFT
numBands = 50;    % number of filters in the auditory spectrogram

% extract audio features using spectrogram - specifically bark spectrum
afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',FFTLength, ...
    'Window',hann(frameSamples,'periodic'), ...
    'OverlapLength',overlapSamples, ...
    'barkSpectrum',true);
setExtractorParams(afe,'barkSpectrum','NumBands',numBands);

%% Process a file from the dataset to get denormalization factor

% apply zero-padding to the audio signal so they are a consistent length of 1 sec
x = read(adsTrain);
numSamples = size(x,1);
numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );
xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

% extract audio features - the output is a Bark spectrum with time across rows
features = extract(afe,xPadded);
[numHops,numFeatures] = size(features);

% determine the denormalization factor to apply for each signal
unNorm = 2/(sum(afe.Window)^2);

%% Feature extraction: read data file, zero pad, then apply spectrogram methods

% Training data: read from datastore, zero-pad, extract spectrogram features
subds = partition(adsTrain,1,1);
XTrain = zeros(numHops,numBands,1,numel(subds.Files));
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
    XTrain(:,:,:,idx) = extract(afe,xPadded);
end
XTrainC{1} = XTrain;
XTrain = cat(4,XTrainC{:});

% extract parameters from training data
[numHops,numBands,numChannels,numSpec] = size(XTrain);

% Scale the features by the window power and then take the log (with small offset)
XTrain = XTrain/unNorm;
epsil = 1e-6;
XTrain = log10(XTrain + epsil);

% Validation data: read from datastore, zero-pad, extract spectrogram features
subds = partition(adsValidation,1,1);
XValidation = zeros(numHops,numBands,1,numel(subds.Files));
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
    XValidation(:,:,:,idx) = extract(afe,xPadded);
end
XValidationC{1} = XValidation;
XValidation = cat(4,XValidationC{:});
XValidation = XValidation/unNorm;
XValidation = log10(XValidation + epsil);

% Isolate the train and validation labels. Remove empty categories.
YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);

% Visualize Data - plot waveforms and play clips for a few examples
specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    
    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

%% now save data as .png images 

% e.g. do 
% tmp_image =  XTrain(:,:,1,i);
% imwrite(tmp_image, ['speechImageData/TrainingData/down/image' num2str(i) '.png']);




% the end



