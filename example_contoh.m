clear all;

%efine the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Data 

%Create an image data store from the raw images 
imdsTrain = imageDatastore('speechImageData\TrainData',...
"IncludeSubfolders",true,"LabelSource","foldernames");

%Create an image validation data store from the validation images 
imdsVal = imageDatastore('speechImageData\ValData',...
"IncludeSubfolders",true,"LabelSource","foldernames");


%Image preprocessing
image_size = [98 50];  
dsTrain = augmentedImageDatastore(image_size, imdsTrain, 'ColorPreprocessing', 'gray2rgb');

dsVal = augmentedImageDatastore(image_size,imdsVal,'ColorPreprocessing', 'gray2rgb');

%Maxpooling limiter
Maxpool_val = 3;

% define constant parameters
num_classes = 12;  % number of classes
filter_size = 3;  % convolutional filter size
timePoolSize = 12; % time pool size
dropoutProb = 0.2; % drop out 

% define the objective function for Bayesian optimization
objective = @(x) trainAndEvaluateNetwork(x, dsTrain, dsVal, imdsVal, num_classes, filter_size, image_size, timePoolSize, dropoutProb, Maxpool_val);

% define the hyperparameter space
hyperparameterSpace = [
    optimizableVariable('num_layers', [5, 6], 'Type', 'integer');
    optimizableVariable('num_filters', [8, 12], 'Type', 'integer')
];

% perform Bayesian optimization
results = bayesopt(objective, hyperparameterSpace, 'MaxObjectiveEvaluations', 5);

% number of models for model averaging
num_models = 5;
% initialize cell array to store the models
nets = cell(1, num_models);

% loop over the number of models
for model_idx = 1:num_models
    % get the best hyperparameters
    num_layers = results.XAtMinEstimatedObjective.num_layers;
    num_filters = results.XAtMinEstimatedObjective.num_filters;

    % define the network layers
    layers = [
        imageInputLayer([image_size 3])
    ];
    
    % add the convolutional layers
    for i = 1:num_layers
        layers = [
            layers
            convolution2dLayer(filter_size, num_filters, Padding="same")
            batchNormalizationLayer
            reluLayer
        ];


        if i <= Maxpool_val
        layers = [
                layers
                maxPooling2dLayer(filter_size, Stride=2, Padding="same")
            ];
        end

        num_filters = num_filters * 2;  % double the number of filters for the next layer
    end
  
    % add the rest of the layers
    layers = [
        layers
        maxPooling2dLayer([timePoolSize,1])
        dropoutLayer(dropoutProb)
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer
    ];

    % training options 
    options = trainingOptions('adam', ...
        "MiniBatchSize",30, ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',15, ...
        'Shuffle','every-epoch', ...
        'ValidationData',dsVal, ...
        'ValidationFrequency',10, ...
        'Verbose',true, ...
        'Plots','training-progress',...
        'ExecutionEnvironment','gpu');

    % train the network
    net = trainNetwork(dsTrain,layers,options);
    
    % store the trained network
    nets{model_idx} = net;
end

% classify the validation output using the trained networks
YPreds = cell(1, num_models);
for model_idx = 1:num_models
    [YPreds{model_idx},probs] = classify(nets{model_idx}, dsVal);
end

lgraph = layerGraph(net);  % Create a layer graph from the network
figure;
plot(lgraph);  % Plot the layer graph

% calculate the mode of the predictions
YPred = mode(cat(5, YPreds{:}), 5);

% extract ground truth labels
YVal = imdsVal.Labels;

% accuracy in percent   
accuracy = 100*sum(YPred == YVal)/numel(YVal);
disp(['The accuracy is: ' num2str(accuracy)])

% plot confusion matrix
figure;
plotconfusion(YVal,YPred)

disp(["Validation Set Accuracy: " num2str(accuracy) "%"]);


% Display sample test images with predicted labels and
% the predicted probabilities of the images having those labels.
idx = randperm(numel(imdsVal.Files),16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsVal,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label)+ ", "+num2str(100*max(probs(idx(i),:)),3)+"%");
end


% Define the function trainAndEvaluateNetwork
function loss = trainAndEvaluateNetwork(params, dsTrain, dsVal, imdsVal, num_classes, filter_size, image_size, timePoolSize, dropoutProb, Maxpool_val)

    num_layers = params.num_layers;
    num_filters = params.num_filters;

    % define the network layers
    layers1 = [
        imageInputLayer([image_size 3])
    ];
    

    % add the convolutional layers
    for i = 1:num_layers
        layers1 = [
            layers1
            convolution2dLayer(filter_size, num_filters, Padding="same")
            batchNormalizationLayer
            reluLayer
        ];


        if i <= Maxpool_val
        layers1 = [
                layers1
                maxPooling2dLayer(filter_size, Stride=2, Padding="same")
            ];
        end

        num_filters = num_filters * 2;  % double the number of filters for the next layer
    end


    % add the rest of the layers
    layers1 = [
        layers1
        maxPooling2dLayer([timePoolSize,1])
        dropoutLayer(dropoutProb)
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer
    ];

    % training options 
    options = trainingOptions('adam', ...
        "MiniBatchSize",30, ...
        'InitialLearnRate',0.001, ...
        'MaxEpochs',15, ...
        'Shuffle','every-epoch', ...
        'ValidationData',dsVal, ...
        'ValidationFrequency',10, ...
        'Verbose',false, ...  % turn off text output
        'Plots','none', ...  % turn off plots
        'ExecutionEnvironment','gpu');

    % train the network
    net = trainNetwork(dsTrain,layers1,options);

    % classify the validation output using the trained network
    YPred = classify(net,dsVal);

    % extract ground truth labels
    YVal = imdsVal.Labels;

    % calculate the loss
    loss = 1 - mean(YPred == YVal);  % loss is 1 - accuracy
end

function outputImage = addGaussianNoise(inputImage)
    noiseSigma = 0.01; % Adjust this to control the noise intensity
    noise = noiseSigma * randn(size(inputImage));
    outputImage = inputImage + noise;
    outputImage = max(min(outputImage, 1), 0); % Clamp values to valid range
end