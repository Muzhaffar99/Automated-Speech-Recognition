originalFolder = {'speechImageData\TrainData', 'speechImageData\ValData'}; 
augmentedFolder = {'speechAug\TrainData', 'speechAug\ValData'};

imds = imageDatastore(originalFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

while hasdata(imds)
    [img, info] = read(imds); % Read an image and its info
    img = im2double(img); % Convert to double
    augmentedImage = addGaussianNoise(img); % Apply noise

    % Construct the path to save the augmented image
    relativePath = strrep(info.Filename, originalFolder, '');
    augmentedPath = fullfile(augmentedFolder, relativePath);

    % Extract directory path
    augmentedDir = fileparts(augmentedPath);

    % Create the folder if it doesn't exist
    if ~exist(augmentedDir, 'dir')
        mkdir(augmentedDir);
    end

    % Save the augmented image
    imwrite(augmentedImage, augmentedPath);
end

function outputImage = addGaussianNoise(inputImage)
    noiseSigma = 0.01; % Standard deviation of Gaussian noise
    noise = noiseSigma * randn(size(inputImage));
    outputImage = inputImage + noise;
    outputImage = max(min(outputImage, 1), 0); % Ensuring values are within the valid range
end