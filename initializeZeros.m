%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

function parameter = initializeZeros(sz,className)
% Initialize a parameter array with zeros.

arguments
    sz                            % Size of the parameter array to be initialized.
    className = 'single'          % Data type of the array, default is 'single'.
end

parameter = zeros(sz,className);  % Create an array of zeros with specified size and type.
parameter = dlarray(parameter);   % Convert the array to a dlarray, making it compatible for deep learning operations.
end
