%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

function [lossG, lossD] = ganLoss(YReal, YGenerated)
lossD = -mean(log(YReal), 'all') - mean(log(1 - YGenerated), 'all');         % Calculate the loss for the discriminator network.
lossG = -mean(log(YGenerated), 'all');                                       % Calculate the loss for the generator network.
end
