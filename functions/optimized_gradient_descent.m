%%% Optimize food wastage through gradient gradient descent
%%
function [optimized_X, J_history] = optimized_gradient_descent(X, y, X_temp, theta, alpha, iterations, lambda)

X = [X; X_temp];
% Optimized output = 0
y = [y; 0];
J_history = zeros(iterations);

m = size(X, 1);
n = size(X, 2);

degrees = round((n - 2) ** (1/3));

for i = 1:iterations
    predict = X * theta;
    % Exclude students from gradient descent
    X(m, 2:3) = X(m, 2:3) - (alpha / m) * (theta(2:3)' * sum(predict - y) + lambda * X(m, 2:3));
    X(m, 3 + degrees:n) = X(m, 3 + degrees:n) - (alpha / m) * (theta(3 + degrees:n)' * sum(predict - y) + lambda * X(m, 3 + degrees:n));
    % TODO: How to add regularization cost
    J_history(i) = (1 / (2 * m)) * ((predict - y)' * (predict - y));
end

optimized_X = X(m, :);
J_history;

end
