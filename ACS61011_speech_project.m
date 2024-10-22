clear all;

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Data 

% create an image data store from the raw images 
imdsTrain = imageDatastore('speechImageData\TrainData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

% create an image validation data store from the validation images 
imdsVal = imageDatastore('speechImageData\ValData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

%%Muzhaffar Maruf Ibrahim - 2301171692

% use the transform function to resize each image
image_size = [98 50];
dsTrain = augmentedImageDatastore(image_size,imdsTrain,'ColorPreprocessing', 'gray2rgb');
dsVal = augmentedImageDatastore(image_size,imdsVal,'ColorPreprocessing', 'gray2rgb');

% define the grid of hyperparameters
num_layers = [2, 3, 4];       % number of layers
num_filters = [32, 64, 128];  % number of filters per layer 
dropout_rate = [0.2, 0.5]; %number of Drop out
num_classes = 12;  % number of classes
timePoolSize = 12; % tuned for good performance here

% Initialize a structure to store results
results = struct('FilterSize', {}, 'NumFilters', {}, 'NumConvBlocks', {}, 'DropoutRate', {}, 'ValidationAccuracy', {});

% initialise best solution
best_accuracy = 0;
best_params = [];

% auxiliary parameters 
aux_params{1} = num_classes;
aux_params{2} = image_size;

% loop over the grid of hyperparameters
% for i = 1:length(num_layers)
%  for j = 1:length(num_filters)
%    for k = 1:length(dropout_rate)

for fs = filterSizes
   for nf = numFilters
     for dr = dropoutRates

    % current hyperparameters
    hyper_params = [num_layers(i), num_filters(j), dropout_rate(k)]

    % create and train model with current hyperparams
    layers = create_model(hyper_params,aux_params);

    % train model
    options = trainingOptions('adam','MiniBatchSize',16,'MaxEpochs',100);

    [net,info] = trainNetwork(dsTrain,layers,options);

    % classify the validation output using the trained network
    [YPred,probs] = classify(net,dsVal);

    % extract ground truth labels
    YVal = imdsVal.Labels;

    % accuracy in percent
    accuracy = 100*sum(YPred == YVal)/numel(YVal);
    disp(['The accuracy is: ' num2str(accuracy)])

    % extract validation accuracy for current model
    accuracy(i,j,k) = info.TrainingAccuracy(end);
    
    % store parameters if they are better than previous
    if accuracy(i,j,k) > best_accuracy
best_accuracy = accuracy;
best_params = hyper_params;
    end
    end
  end
end

% define a function to create a model
function layers = create_model(hyper_params,aux_params)

% unpack hyperparameter values under test
num_layers = hyper_params(1);
num_filters = hyper_params(2);
dropout_rate = hyper_params(3);

% unpack auxiliary parameters needed to build network
num_classes = aux_params{1};
image_size = aux_params{2};

% create input layer
layers = [
imageInputLayer([image_size 3])
];


% create blocks of conv -> batch norm -> relu -> max pool layers
for i = 1:num_layers
    layers = [layers
    convolution2dLayer(3,num_filters,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([12,1])
    ];
end

% output layers
layers = [layers
dropoutLayer(dropout_rate) 
fullyConnectedLayer(num_classes)
softmaxLayer
classificationLayer];
end




