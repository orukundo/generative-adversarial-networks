%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

function [lossG, lossD, gradientsG, gradientsD, stateG, scoreG, scoreD] = modelLoss(netG, netD, X, Z, flipProb)
YReal = forward(netD, X);                                             % Calculate the predictions for real data with the discriminator network.

% Calculate the predictions for generated data with the discriminator network.
[XGenerated, stateG] = forward(netG, Z);
YGenerated = forward(netD, XGenerated);
scoreD = (mean(YReal, 'all') + mean(1 - YGenerated, 'all')) / 2;      % Calculate the score of the discriminator.
scoreG = mean(YGenerated, 'all');                                     % Calculate the score of the generator.

% Randomly flip the labels of the real images.
numObservations = size(YReal, 4);
idx = rand(1, numObservations) < flipProb;
YReal(:,:,:,idx) = 1 - YReal(:,:,:,idx);
[lossG, lossD] = ganLoss(YReal, YGenerated);                          % Calculate the GAN loss.

% Ensure lossG and lossD are scalars by using mean if necessary
lossG = mean(lossG, 'all');
lossD = mean(lossD, 'all');

% For each network, calculate the gradients with respect to the loss.
gradientsG = dlgradient(lossG, netG.Learnables, 'RetainData', true);
gradientsD = dlgradient(lossD, netD.Learnables);
end
