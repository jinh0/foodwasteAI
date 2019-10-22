%%% Optimization of Food Waste v0.1
%%% TO DO:
%%%     + Gradient descent on X_training to minimize wastage

% Access path to functions
addpath('functions')

% Load data and separate into X and y
data = load('foodwaste2017/dinner/total_dinner.txt');

X = data(:,1:3);
y = data(:, 4);

num = input('Number of polynomial variables (default 3): ');

% Add polynomial variables
if num > 1
    X = polynomial_features1(X, num);
end

% Split data
[X_training, X_test, y_training, y_test] = splitdata(X, y);

% More useful variables
m = size(X_training, 1);
n = size(X_training, 2);

mu = mean(X_training);
sigma = std(X_training);

% Normalize features
X_training = normalize_features(X_training, mu, sigma);
X_training = [ones(size(X_training, 1), 1), X_training];

fprintf('Gradient descent applied.\n');

% Gradient descent
theta = rand(n + 1, 1);
alpha = 0.01;
iterations = 5000;

lambda = 0.1;

[theta, J_history] = regularized_gradient_descent(X_training, y_training, theta, alpha, iterations, lambda);

plot(1:iterations, J_history)
xlabel('Number of iterations');
ylabel('Cost function');

J = cost_function(X_training, y_training, theta, lambda);

fprintf('\nCost: %f\n\n', J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Optimize the amount of rice and dal supplied for specified number of students
% Gradient descent on X_TRAINING
students = input('Number of students: ');

X_temp = [mu(1:2), students];
X_temp = polynomial_features1(X_temp, num);
X_temp = [1, normalize_features(X_temp, mu, sigma)];

[optimized_X, J_history] = optimized_gradient_descent(X_training, y_training, X_temp, theta, alpha, iterations, lambda);

optimized_X * theta

[1, normalize_features(mu, mu, sigma)] * theta

optimized_X = denormalize_features(optimized_X(1, 2:n + 1), mu, sigma);
optimized_X(1:4)

mu(1:4)
