function [theta, J_history] = regularized_gradient_descent(X, y, theta, alpha, iterations, lambda)

m = size(X, 1);
n = size(X, 2);

J_history = [];

for i = 1:iterations
    theta(1) = theta(1) - (alpha / m) * ((X(:,1))' * (X * theta - y));
    theta(2:n) = theta(2:n) .- (alpha / m) * ((X(:,2:n))' * (X * theta - y) .+ lambda .* theta(2:n));
    J_history = [J_history; ((X * theta - y)' * (X * theta - y) + lambda * ((theta(2:n))' * theta(2:n))) / (2 * m)];
end

end