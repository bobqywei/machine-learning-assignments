function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================

initial_theta = all_theta(1,:);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for i = 1: num_labels
    all_theta(i, :) = fmincg(@(t)lrCostFunction(t, X, y == i, lambda), initial_theta', options);
end

% =========================================================================


end
