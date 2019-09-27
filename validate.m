% Program to check validation and debug through iterations
% TO DO:
%   Error vs Number of Examples
fprintf('Validation and Debugging\n\n');

data = load('foodwaste2017/dinner/total_dinner.txt');

% Initialization of useful variables
X = data(:,1:3);
y = data(:,4);

degrees = input('Degree of polynomial variables (default 1): ');

% Add polynomial variables
if degrees > 1
    X = polynomial_features(X, degrees);
end

% Split data into training and test data
[X_training, X_test, y_training, y_test] = splitdata(X, y);

m = size(X_training, 1);
m_test = size(X_test, 1);
n = size(X_training, 2);

mu = mean(X_training);
mu_test = mean(X_test);
sigma = std(X_training);
sigma_test = std(X_test);

% Normalize features
X_training = normalize_features(X_training, mu, sigma);
X_training = [ones(m, 1), X_training];

X_test = normalize_features(X_test, mu_test, sigma_test);
X_test = [ones(m_test, 1), X_test];

alpha = 0.01;
iterations = 5000;
lambda = 1;

% Store cost function history
training_errors = zeros(m, 1);
test_errors = zeros(m, 1);

for debug_m = 1:m
    theta = rand(n + 1, 1);
    X_current = X_training(1:debug_m, :);
    y_current = y_training(1:debug_m);
    [theta, J_history] = regularized_gradient_descent(X_current, y_current, theta, alpha, iterations, lambda);

    training_errors(debug_m) = cost_function(X_current, y_current, theta, lambda);
    test_errors(debug_m) = cost_function(X_test, y_test, theta, lambda);
end

fprintf('Test data vs Training data cost\n');
plot(1:m, training_errors, 1:m, test_errors);

fprintf('Cost for training data: %f\nCost for test data: %f\n', training_errors(m), test_errors(m));

fprintf('\nEnd of program.\n');
