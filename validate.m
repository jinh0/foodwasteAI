% Validation Program
%   Creates visualizations and learning graphs
%   Current list of graphs:
%       + Number of features vs Cost
%       + Number of examples vs Cost
%       + Lambda vs Cost

% Access path to functions
addpath('functions')

fprintf('Validation and Debugging\n\n');

% Load data and separate into X and y
data = load('foodwaste2017/dinner/total_dinner.txt');
X = data(:,1:3);
y = data(:,4);

degrees = input('Degree of polynomial variables (default 1): ');

X = X(randperm(size(X, 1)), :);

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

%%% Number of features vs Cost

% Maximum number of features, calculated as just below the number of features created in polynomial_features();
features = degrees ^ 3;

% Store cost function history
training_errors = zeros(features - 3, 1);
test_errors = zeros(features - 3, 1);

for num = 4:features
    theta = rand(num, 1);
    [theta, J_history] = regularized_gradient_descent(X_training(:, 1:num), y_training, theta, alpha, iterations, lambda);

    training_errors(num - 3) = cost_function(X_training(:, 1:num), y_training, theta, lambda);
    test_errors(num - 3) = cost_function(X_test(:, 1:num), y_test, theta, lambda);
end

fprintf('Training and test cost against number of features\n\n');
plot(4:features, training_errors, 4:features, test_errors);
xlabel('Number of features');
ylabel('Cost');

fprintf('Program paused. Press enter to continue.\n');
pause;

%%% Number of examples vs Cost

% Reinitalize cost function history
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

fprintf('Training and test cost against number of examples\n\n');
plot(1:m, training_errors, 1:m, test_errors);
xlabel('Number of examples');
ylabel('Cost');

fprintf('Program paused. Press enter to continue.\n');
pause;

%%% Lambda vs Cost

% Reinitalize cost function history
training_errors = zeros(100, 1);
test_errors = zeros(100, 1);

for l = 1:100
    theta = rand(n + 1, 1);
    [theta, J_history] = regularized_gradient_descent(X_training, y_training, theta, alpha, iterations, l * 0.1);

    training_errors(l) = cost_function(X_training, y_training, theta, l * 0.1);
    test_errors(l) = cost_function(X_test, y_test, theta, l * 0.1);
end

fprintf('Training and test cost against number of examples\n\n');
plot(1:100, training_errors, 1:100, test_errors, '-', 2);
xlabel('Lambda, scaled up by 10x');
ylabel('Cost');

fprintf('Cost for training data: %f\nCost for test data: %f\n', training_errors(m), test_errors(m));

fprintf('\nEnd of program.\n');
