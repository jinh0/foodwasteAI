function [theta, J_history, theta_history] = gradient_descent(X, y, theta, alpha, iterations)

m = size(X, 1);

J_history = [];
theta_history = [];

for i = 1:iterations
    theta(2:3) = theta(2:3) .- ((alpha / m) * (X(:, 2:3))' * (X * theta - y));
    J_history = [J_history; (X * theta - y)' * (X * theta - y) / (2 * m)];
    theta_history = [theta_history; [theta(2), theta(3)]];
end

end
