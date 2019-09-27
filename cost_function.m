function J = cost_function(X, y, theta, lambda)

m = size(X, 1);
n = size(X, 2);

J = ((X * theta - y)' * (X * theta - y) + lambda * ((theta(2:n))' * theta(2:n))) / (2 * m);
