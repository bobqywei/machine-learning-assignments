function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Cost function and regularization

X = [ones(m, 1), X];
hiddenLayerActivations = sigmoid(X * Theta1'); % [5000x401]*[401*25]

hiddenLayerActivations = [ones(m, 1), hiddenLayerActivations];
predictions = sigmoid(hiddenLayerActivations * Theta2'); % [5000x26]*[26x10] = [5000x10]

Y = zeros(m, num_labels);
for i = 1:m
    Y(i, y(i)) = 1;
end

for i = 1:m
    for k = 1:num_labels
        J += (-Y(i, k) * log(predictions(i, k)) - (1 - Y(i, k)) * log(1 - predictions(i, k)));
    end
end

J /= m;


regTerm = 0;

for j = 1:size(Theta1, 1)
    for k = 2:size(Theta1, 2)
        regTerm += (Theta1(j, k) ^ 2);
    end
end

for j = 1:size(Theta2, 1)
    for k = 2:size(Theta2, 2)
        regTerm += (Theta2(j, k) ^ 2);
    end
end

regTerm *= (lambda / (2*m));

J += regTerm;

%
%Backpropagation
%
for i = 1:m
    a_1 = X(i:i, :);
    a_1 = a_1';

    z_2 = Theta1 * a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1 ; a_2];

    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);

    y_vec = zeros(1, num_labels);
    delta_3 = zeros(1, num_labels);
    delta_2 = zeros(columns(Theta1), 1);

    y_vec(y(i)) = 1;
    for j = 1:num_labels
        delta_3(j) = a_3(j:j, :) - y_vec(j);
    end
    delta_3 = delta_3';

    a_2_gradient = [1 ; sigmoidGradient(z_2)];
    delta_2 = (Theta2' * delta_3) .* a_2_gradient;
    delta_2 = delta_2(2:end);

    Theta1_grad += delta_2 * a_1';
    Theta2_grad += delta_3 * a_2';
end
Theta1_grad /= m;
Theta2_grad /= m;

% Back propagation with regularization

for i = 1:rows(Theta1)
    for j = 2:columns(Theta1)
        Theta1_grad(i, j) += Theta1(i, j) * (lambda / m); 
    end
end

for i = 1:rows(Theta2)
    for j = 2:columns(Theta2)
        Theta2_grad(i, j) += Theta2(i, j) * (lambda / m); 
    end
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
