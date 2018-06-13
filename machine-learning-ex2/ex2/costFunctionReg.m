function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
H = zeros(m);

for i = 1:m
    h = sigmoid(X(i, :) * theta);
    H(i) = h;
    J += (-y(i) * log(h) - (1 - y(i)) * log(1 - h));
end
J /= m;

reg_total = 0;
for j = 2:size(theta)
    reg_total += theta(j) ^ 2;
end
J += (lambda / (2 * m)) * reg_total;



for j = 1:size(theta)
    for i = 1:m
        grad(j) += (H(i) - y(i)) * X(i, j);
    end
    grad(j) /= m;

    if (j > 1)
        grad(j) += (lambda / m) * theta(j);
    endif
end

% =============================================================

end
