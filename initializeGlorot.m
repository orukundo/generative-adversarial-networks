%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

function weights = initializeGlorot(sz,numOut,numIn,className)
% Initialize weights using Glorot (Xavier) initialization.

arguments
    sz                    % Size of the weight matrix.
    numOut                % Number of output units in the layer.
    numIn                 % Number of input units to the layer.
    className = 'single'  % Data type of the weights, default is 'single'.
end

Z = 2*rand(sz,className) - 1;        % Generate random values in [-1, 1] with specified size and type.
bound = sqrt(6 / (numIn + numOut));  % Compute the Glorot initialization boundary.
weights = bound * Z;                 % Scale random values by the boundary to get initial weights.
weights = dlarray(weights);          % Convert weights to a differentiable array for deep learning.
end
