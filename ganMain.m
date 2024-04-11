%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

%% Load Training Data 
imageFolder = fullfile('C:\...\RealImages');                                            % Read folder containing real images of interest.
imds = imageDatastore(imageFolder,IncludeSubfolders=true);                              % Create an image datastore containing the pictures of interest.
augmenter = imageDataAugmenter(RandXReflection=true);                                   % Augment the data to include random horizontal flipping 
augimds = augmentedImageDatastore([256 256],imds,DataAugmentation=augmenter);           %...and resize the images to have the desized size (e.g., 256 x 256).

%% Define Generator Network
filterSize = 5;                                                                      % Sets the height and width of filters in transposed convolutional layers to 5x5.
numFilters = 128;                                                                    % Initial number of filters in the first transposed convolutional layer.
numLatentInputs = 100;                                                               % Size of the input latent vector, a 100-dimensional vector from latent space.
projectionSize = [4 4 1024];                                                         % Reshapes the latent vector into a 4x4x1024 tensor as the starting point for image generation.
numOutputChannels = 1;                                                               % Number of channels in the output image, 1 for grayscale images.

layersGenerator = [
    featureInputLayer(numLatentInputs)                                               % Accepts input latent vector of specified size.
    projectAndReshapeLayer(projectionSize)                                           % Projects and reshapes latent vector into a predefined tensor shape.
    transposedConv2dLayer(filterSize, 4*numFilters, Stride=2, Cropping="same")       % First layer to upscale and transform the input with transposed convolution.
    batchNormalizationLayer                                                          % Normalizes the output to stabilize training.
    reluLayer                                                                        % Applies ReLU activation function for non-linearity.
    transposedConv2dLayer(filterSize, 2*numFilters, Stride=2, Cropping="same")       % Upscales further while reducing depth.
    batchNormalizationLayer                                                          % Another batch normalization for the next layer's output.
    reluLayer                                                                        % ReLU activation to introduce non-linearity.
    transposedConv2dLayer(filterSize, numFilters, Stride=2, Cropping="same")         % Continues to upscale and reduce filter depth.
    batchNormalizationLayer                                                          % Batch normalization to standardize outputs before activation.
    reluLayer                                                                        % ReLU activation for non-linear transformation.
    transposedConv2dLayer(filterSize, 0.5*numFilters, Stride=2, Cropping="same")     % Further upscales and decreases depth for finer details.
    batchNormalizationLayer                                                          % Applies normalization to this layer's output.
    reluLayer                                                                        % Another ReLU layer for non-linearity.
    transposedConv2dLayer(filterSize, 0.25*numFilters, Stride=2, Cropping="same")    % Added layer increases image resolution with reduced depth.
    batchNormalizationLayer                                                          % Normalizes outputs from the previous transposed convolution layer.
    reluLayer                                                                        % Applies ReLU for non-linear transformations.
    transposedConv2dLayer(filterSize, numOutputChannels, Stride=2, Cropping="same")  % Final layer to achieve target image size and depth (channels).
    tanhLayer                                                                        % Tanh activation to scale output pixel values to [-1, 1].
];

netG = dlnetwork(layersGenerator);                                                   % Wraps the generator layers into a differentiable network object for training.

%% Define Discriminator Network
dropoutProb = 0.5;                                                            % Sets dropout probability to 50% to reduce overfitting.
numFilters = 128;                                                              % Initial number of filters for the first convolutional layer.
scale = 0.2;                                                                  % Slope for the negative part of the LeakyReLU activation function.
inputSize = [256 256 1];                                                      % Specifies input image size as 256x256 pixels with 1 channel (grayscale).
filterSize = 5;                                                               % Size of convolution filters is set to 5x5.

layersDiscriminator = [
    imageInputLayer(inputSize, Normalization="none")                          % Input layer accepting 256x256 grayscale images, without applying normalization.
    dropoutLayer(dropoutProb)                                                 % Dropout layer applied to the input images to prevent overfitting.
    convolution2dLayer(filterSize, 0.25*numFilters, Stride=2, Padding="same") % First convolutional layer, using a quarter of the base number of filters.
    leakyReluLayer(scale)                                                     % LeakyReLU activation function allows a small, non-zero gradient when the unit is not active.
    convolution2dLayer(filterSize, 0.5*numFilters, Stride=2, Padding="same")  % Second convolution layer, using half of the base number of filters.
    batchNormalizationLayer                                                   % Batch normalization layer to normalize the activations of the previous layer.
    leakyReluLayer(scale)                                                     % Another LeakyReLU layer for non-linear activation.
    convolution2dLayer(filterSize, numFilters, Stride=2, Padding="same")      % Third convolution layer with the base number of filters.
    batchNormalizationLayer                                                   % Applies batch normalization again to stabilize the learning.
    leakyReluLayer(scale)                                                     % LeakyReLU activation function for introducing non-linearity.
    convolution2dLayer(filterSize, 2*numFilters, Stride=2, Padding="same")    % Fourth convolution layer, doubling the base number of filters.
    batchNormalizationLayer                                                   % Another batch normalization layer to normalize outputs.
    leakyReluLayer(scale)                                                     % LeakyReLU layer to maintain non-linearity.
    convolution2dLayer(filterSize, 4*numFilters, Stride=2, Padding="same")    % Fifth convolution layer, quadrupling the base number of filters for detailed feature extraction.
    batchNormalizationLayer                                                   % Batch normalization layer for output stabilization.
    leakyReluLayer(scale)                                                     % LeakyReLU activation function for non-linear processing.
    convolution2dLayer(4, 1, Padding="same")                                  % Final convolutional layer to consolidate features into a single output prediction.
    sigmoidLayer                                                              % Sigmoid activation function to squash the output to a probability [0, 1].
    ];

netD = dlnetwork(layersDiscriminator);                                        % Wraps the discriminator layers into a differentiable network object.

%% Define Model Loss Functions, GAN Loss Function and Scores, Mini-Batch Preprocessing Function
% function [lossG,lossD,gradientsG,gradientsD,stateG,scoreG,scoreD] = modelLoss(netG,netD,X,Z,flipProb)
% function [lossG,lossD] = ganLoss(YReal,YGenerated)
% function X = preprocessMiniBatch(data)

%% Specify Training Options
numEpochs = 5000;                              % Number of full passes through the training dataset.
miniBatchSize = 2;                             % Number of samples per mini-batch for gradient estimation.
learnRate = 0.0002;                            % Learning rate for the optimizer, controlling the step size during weight updates.
gradientDecayFactor = 0.5;                     % Momentum factor, helping to accelerate gradients vectors in the right directions.
squaredGradientDecayFactor = 0.999;            % Factor for the moving average of the squared gradient values, used in optimizers like Adam.
flipProb = 0.35;                               % Probability of flipping the labels when training the discriminator to add noise to the training process.
validationFrequency = 100;                     % How often to check the validation set for monitoring overfitting and underfitting.

%% Train Model
augimds.MiniBatchSize = miniBatchSize;        % Set mini-batch size for augmented image datastore.
mbq = minibatchqueue(augimds, MiniBatchSize=miniBatchSize, PartialMiniBatch="discard", MiniBatchFcn=@preprocessMiniBatch, MiniBatchFormat="SSCB"); % Create a minibatch queue from augmented data, specifying how to handle partial batches, preprocessing function, and data format.

% Initialize the parameters for Adam optimization.
trailingAvgG = [];
trailingAvgSqG = [];
trailingAvg = [];
trailingAvgSqD = [];

% Create an array of held-out random values.
numValidationImages = 4;                                            % Set number of images for validation.
ZValidation = randn(numLatentInputs,numValidationImages,"single");  % Generate random validation latent vectors (with `numLatentInputs` as the size of each latent vector & `numValidationImages` as the number of latent vectors to generate)
ZValidation = dlarray(ZValidation,"CB");                            % Convert the data to dlarray objects and specify the format "CB" (channel, batch).

% For GPU training, convert the data to gpuArray objects.
if canUseGPU
    ZValidation = gpuArray(ZValidation);
end

% To track the scores for the generator and discriminator, use a trainingProgressMonitor object. 
numObservationsTrain = numel(imds.Files);
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

% Initialize the TrainingProgressMonitor object.
monitor = trainingProgressMonitor(Metrics=["GeneratorScore","DiscriminatorScore"], Info=["Epoch","Iteration"], XLabel="Iteration");
groupSubPlot(monitor,Score=["GeneratorScore","DiscriminatorScore"])

% Train the GAN. For each epoch, shuffle the datastore and loop over mini-batches of data.
epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Reset and shuffle datastore.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        % Read mini-batch of data.
        X = next(mbq);
        % Generate latent inputs for the generator network. Convert to dlarray and specify the format "CB" (channel, batch). If a GPU is available, then convert latent inputs to gpuArray.
        Z = randn(numLatentInputs,miniBatchSize,"single");
        Z = dlarray(Z,"CB");
        if canUseGPU
            Z = gpuArray(Z);
        end
        % Evaluate the gradients of the loss with respect to the learnable parameters, the generator state, and the network scores using dlfeval and the modelLoss function.
        [~,~,gradientsG,gradientsD,stateG,scoreG,scoreD] = dlfeval(@modelLoss,netG,netD,X,Z,flipProb);
        netG.State = stateG;
        % Update the discriminator network parameters.
        [netD,trailingAvg,trailingAvgSqD] = adamupdate(netD, gradientsD, trailingAvg, trailingAvgSqD, iteration, learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the generator network parameters.
        [netG,trailingAvgG,trailingAvgSqG] = adamupdate(netG, gradientsG, trailingAvgG, trailingAvgSqG, iteration, learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % % Every validationFrequency iterations, display batch of generated images using the held-out generator input.
        % if mod(iteration,validationFrequency) == 0 || iteration == 1
        %     XGeneratedValidation = predict(netG,ZValidation);   % Generate images using the held-out generator input.
        % 
        %     % Tile and rescale the images in the range [0 1].
        %     I = imtile(extractdata(XGeneratedValidation));
        %     I = rescale(I);
        % 
        %     % Display the images.
        %     imshow(I,[])
        %     xticklabels([]);
        %     yticklabels([]);
        %     title("Generated Images");
        % end

        % Update the training progress monitor.
        recordMetrics(monitor,iteration, GeneratorScore=scoreG, DiscriminatorScore=scoreD);
        updateInfo(monitor,Epoch=epoch,Iteration=iteration);
        progressValue = 100 * iteration / numIterations;
        progressValue = max(0, min(100, progressValue)); 
        monitor.Progress = progressValue;

        save('netG_pretrained.mat', 'netG');  % Save the trained generator network netG to a file named netG_pretrained.mat in the current working directory.
    end
end

%% Generate New Images
% load('netG_pretrained.mat', 'netG');                            % load the netG network from netG_pretrained.mat back into the workspace further operations.
% numLatentInputs = 100;                                          % Size of the input latent vector, a 100-dimensional vector from latent space.
numObservations = 4;                                              % Choose any number of images you want to GAN to fake or generate using the pre-trained 'netG_pretrained.mat'.
ZNew = randn(numLatentInputs,numObservations,"single");
ZNew = dlarray(ZNew,"CB");

% If a GPU is available, then convert the latent vectors to gpuArray.
if canUseGPU
    ZNew = gpuArray(ZNew);
end

% Generate new images using the predict function with the generator and the input data.
XGeneratedNew = predict(netG,ZNew);

% Define the output folder for synthetic images by GAN
outputFolder = 'C:\...\FakeImages';

% Ensure the output folder exists, create if it does not
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

XGeneratedNew = extractdata(XGeneratedNew);                % Extract and process generated images from the dlarray
XGeneratedNew = rescale(XGeneratedNew);                    % Rescale images to the range [0, 1] for display or saving
numImages = size(XGeneratedNew, numObservations);          % Generate new images using the predict function with the generator and the input data

for i = 1:numImages
    img = XGeneratedNew(:,:,:,i);                           % Select the i-th image
    if size(img, 3) == 1                                    % Ensure it is a grayscale image
        img = squeeze(img);                                 % Remove singleton dimension for grayscale images
    end
    fileName = sprintf('gan_generated_image_%d.png', i);    % Define the file name
    fullPath = fullfile(outputFolder, fileName);            % Full file path
    imwrite(img, fullPath);                                 % Save the image
end


