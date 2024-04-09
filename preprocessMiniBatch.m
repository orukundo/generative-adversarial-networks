%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

function X = preprocessMiniBatch(data)
X = cat(4,data{:});                             % Concatenate mini-batch
X = rescale(X,-1,1,InputMin=0,InputMax=255);    % Rescale the images in the range [-1 1].
end