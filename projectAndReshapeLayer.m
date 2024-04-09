%% Synthetic Biomedical Image Generation with GANs
% Olivier Rukundo, Ph.D. - orukundo@gmail.com - 2024-04-09
% This script is a modification of the example provided by MathWorks on training generative adversarial networks.
% Original source: https://se.mathworks.com/help/deeplearning/ug/train-generative-adversarial-network.html

classdef projectAndReshapeLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable

    properties
        % Layer properties.
        OutputSize
    end

    properties (Learnable)
        % Layer learnable parameters.
        Weights
        Bias
    end

    methods
        function layer = projectAndReshapeLayer(outputSize,NameValueArgs)
            % Parse input arguments.
            arguments
                outputSize
                NameValueArgs.Name = "";
            end

            % Set layer name.
            name = NameValueArgs.Name;
            layer.Name = name;

            % Set layer description.
            layer.Description = "Project and reshape to size " + join(string(outputSize));

            % Set layer type.
            layer.Type = "Project and Reshape";

            % Set output size.
            layer.OutputSize = outputSize;
        end

        function layer = initialize(layer,layout)
            % Layer output size.
            outputSize = layer.OutputSize;

            % Initialize fully connect weights.
            if isempty(layer.Weights)

                % Find number of channels.
                idx = finddim(layout,"C");
                numChannels = layout.Size(idx);

                % Initialize using Glorot.
                sz = [prod(outputSize) numChannels];
                numOut = prod(outputSize);
                numIn = numChannels;
                layer.Weights = initializeGlorot(sz,numOut,numIn);
            end

            % Initialize fully connect bias.
            if isempty(layer.Bias)

                % Initialize with zeros.
                layer.Bias = initializeZeros([prod(outputSize) 1]);
            end
        end

        function Z = predict(layer, X)
            % Fully connect.
            weights = layer.Weights;
            bias = layer.Bias;
            X = fullyconnect(X,weights,bias);

            % Reshape.
            outputSize = layer.OutputSize;
            Z = reshape(X,outputSize(1),outputSize(2),outputSize(3),[]);
            Z = dlarray(Z,"SSCB");
        end
    end
end