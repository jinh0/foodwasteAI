function [theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)

m = size(X, 1);

J_history = [];

for i = 1:iterations
    theta = theta .- ((alpha / m) * X' * (X * theta - y));
    J_history = [J_history; (X * theta - y)' * (X * theta - y) / (2 * m)];
end

end
