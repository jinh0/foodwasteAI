% Student Prediction using rice and dal data v0.4
%   Split data + Synthetic Data Method 2
%   Regularization
%   Automated polynomial variables + output, Linear Regression Model

fprintf('Student Prediction v0.4\n');
fprintf('Using rice and dal data + automated polynomial features,\nregularization, and synthetic + split data\n\n');

data = load('foodwaste2017/dinner/total_dinner.txt');

% Initialization of useful variables
X = data(:,1:3);
y = data(:,4);

degrees = input('Degree of polynomial variables (default 1): ');

%num = input('Value of parameter: ');

% Add polynomial variables
if num > 4
    X = polynomial_features(X, degrees, num);
end


% Split data
[X_training, X_test, y_training, y_test] = splitdata(X, y);

% More useful variables
m = size(X_training, 1);
n = size(X_training, 2);

mu = mean(X_training);
sigma = std(X_training);

% Initial visualization
scatter3(X_training(:, 1), X_training(:, 2), y_training)
xlabel('Rice');
ylabel('Dal');
zlabel('Wastage');

%in = input('Add fake data? 0 for no, 1 for yes: ');

%if in == 1
%    [X_fake, y_fake] = generate_synthetic_data(mu, sigma, y_training, n);
%
%    X_training = [X_training; X_fake];
%    y_training = [y_training; y_fake];
%endif

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nFeatures normalized.\n\n');

% Normalize features
X_training = normalize_features(X_training, mu, sigma);
X_training = [ones(size(X_training, 1), 1), X_training];

fprintf('Gradient descent applied.\n\n');

% Gradient descent
theta = rand(n + 1, 1);
alpha = 0.01;
iterations = 5000;

[theta, J_history] = gradient_descent(X_training, y_training, theta, alpha, iterations);

plot(1:iterations, J_history);
xlabel('Number of Iterations');
ylabel('Cost');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nWith %0.1f iterations and learning rate of %0.2f:\n\n', iterations, alpha);
fprintf('Theta:\n');
fprintf('Intecept = %f\n', theta(1));
fprintf('Rice = %f\n', theta(2));
fprintf('Dal = %f\n', theta(3));
fprintf('Students = %f\n', theta(4));
%fprintf('Rice X_training Dal = %f\n', theta(5));
%fprintf('Rice Squared = %f\n', theta(6));
%fprintf('Dal Squared = %f\n', theta(7));

J = sum(((X_training * theta) - y_training) .^ 2) / (2 * m);
fprintf('\nCost: %f\n\n', J);

lambda = 1;
[theta, J_history] = regularized_gradient_descent(X_training, y_training, theta, alpha, iterations, lambda);

fprintf('With %0.1f iterations and learning rate of %0.2f with regularization, %0.2f:\n\n', iterations, alpha, lambda);
fprintf('Theta:\n');
fprintf('Intecept = %f\n', theta(1));
fprintf('Rice = %f\n', theta(2));
fprintf('Dal = %f\n', theta(3));
fprintf('Students = %f\n', theta(4));
%fprintf('Rice x Dal = %f\n', theta(5));
%fprintf('Rice Squared = %f\n', theta(6));
%fprintf('Dal Squared = %f\n', theta(7));

J = (sum(((X_training * theta) - y_training) .^ 2) + lambda * sum(theta(2:n) .^ 2)) / (2 * m);
fprintf('\nCost: %f\n\n', J);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nPredictions against TRAINING Data Visualization:\n');
fprintf('BLUE: training data points\nRED: predicted points\n\n');

scatter(X_training(:, 2), y_training);
xlabel('Rice');
ylabel('Wastage');
title('Training Data against Predicted Points when Degrees = 1');

hold on
scatter(X_training(:, 2), X_training * theta, 'r');
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nPredictions against TEST Data Visualization:\n');

mu_test = mean(X_test);
sigma_test = std(X_test);
X_test = normalize_features(X_test, mu_test, sigma_test);

X_test = [ones(size(X_test, 1), 1), X_test];

scatter3(X_test(:, 3), X_test(:, 2), y_test)
xlabel('Dal');
ylabel('Rice');
zlabel('Wastage');

hold on
scatter3(X_test(:, 3), X_test(:, 2), X_test * theta, 'r')
hold off

J = ((sum(((X_test * theta) - y_test) .^ 2)) + lambda * sum(theta(2:n) .^ 2)) / (2 * length(y_test));
fprintf('Cost: %f\n\n', J);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nPredictions for Specific Values:\n');

random = floor(rand(1) * m);
X_predict = X_training(random, :);
y_predict = y_training(random);
prediction = X_predict * theta;

fprintf('Prediction %f for real value in TRAINING data (%f).\n', prediction, y_predict);

random = floor(rand(1) * size(X_test, 1));
X_predict = X_test(random, :);
y_predict = y_test(random);
prediction = X_predict * theta;

fprintf('Prediction %f for real value in TEST data (%f).\n', prediction, y_predict);

fprintf('\nEnd of program.\n');
