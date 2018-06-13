function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
row = rows(z);
cols = columns(z);

if (row > 1 && cols > 1)
    for i = 1:row
        for j = 1:cols
            g(i, j) = 1 / (1 + exp(-z(i, j)));
        end
    end
elseif (row > 1)
    for i = 1:row
        g(i) = 1 / (1 + exp(-z(i)));
    end
elseif (cols > 1)
    for j = 1:cols
        g(1, j) = 1 / (1 + exp(-z(1, j)));
    end
else
    g = 1 / (1 + exp(-z));
endif



% =============================================================

end
