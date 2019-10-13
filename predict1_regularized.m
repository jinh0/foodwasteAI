% Student Prediction using rice and dal data
% Regularization
% Polynomial variables + output, Linear Regression Model

% Access path to functions
addpath('/functions')

fprintf('Student Prediction v0.3\n');
fprintf('Using rice and dal data + polynomial features\nand Regularization\n\n');

data = load('foodwaste2017/dinner/total_dinner.txt');

% Initialization of Useful Variables
X = data(:,1:3);
y = data(:,4);

% Add rice x dal variable
X = [X, X(:,1) .* X(:,2)];
% Add rice^2 variable
X = [X, X(:,1) .^ 2];
% Add dal^2 variable
X = [X, X(:,2) .^ 2];

% More useful variables
m = size(X, 1);
n = size(X, 2);

mu = mean(X);
sigma = std(X);

% Initial Visualization
scatter3(X(:,1), X(:,2), y)
xlabel('Rice');
ylabel('Dal');
zlabel('Wastage');

in = input('Add fake data? 0 for no, 1 for yes: ');

if in == 1
    [X_fake, y_fake] = generate_synthetic_data(mu, sigma, y, n);

    X = [X; X_fake];
    y = [y; y_fake];
endif

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Features normalized.\n\n');

% Normalize Features
X = normalize_features(X, mu, sigma);
X = [ones(size(X, 1), 1), X];

scatter3(X(:,2), X(:,4), y)
xlabel('Rice');
ylabel('Students');
zlabel('Wastage');

fprintf('Gradient descent applied.\n\n');

% Gradient Descent
theta = rand(n + 1, 1);
alpha = 0.01;
iterations = 10000;

[theta, J_history] = gradient_descent(X, y, theta, alpha, iterations);

fprintf('With %0.1f iterations and learning rate of %0.2f:\n\n', iterations, alpha);
fprintf('Theta:\n');
fprintf('Intecept = %f\n', theta(1));
fprintf('Rice = %f\n', theta(2));
fprintf('Dal = %f\n', theta(3));
fprintf('Students = %f\n', theta(4));
fprintf('Rice x Dal = %f\n', theta(5));
fprintf('Rice Squared = %f\n', theta(6));
fprintf('Dal Squared = %f\n', theta(7));

lambda = 3;
[theta, J_history] = regularized_gradient_descent(X, y, theta, alpha, iterations, lambda);

J = sum(((X * theta) - y) .^ 2) / (2 * m);
fprintf('\nCost: %f\n\n', J);

fprintf('With %0.1f iterations and learning rate of %0.2f with regularization, %0.2f:\n\n', iterations, alpha, lambda);
fprintf('Theta:\n');
fprintf('Intecept = %f\n', theta(1));
fprintf('Rice = %f\n', theta(2));
fprintf('Dal = %f\n', theta(3));
fprintf('Students = %f\n', theta(4));
fprintf('Rice x Dal = %f\n', theta(5));
fprintf('Rice Squared = %f\n', theta(6));
fprintf('Dal Squared = %f\n', theta(7));

J = sum(((X * theta) - y) .^ 2) / (2 * m);
fprintf('\nCost: %f\n\n', J);

plot(1:iterations, J_history);
xlabel('Number of Iterations');
ylabel('Cost');

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Predictions against Real Data Visualization:\n\n');
scatter3(X(:,3), X(:,2), y)
xlabel('Dal');
ylabel('Rice');
zlabel('Wastage');

hold on;
scatter3(X(:,3), X(:,2), X * theta, 'r')
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('Rice Squared and Dal Squared\n\n');

scatter3(X(:,6), X(:,7), y)
xlabel('Rice Squared');
ylabel('Dal Squared');
zlabel('Wastage');

hold on;
scatter3(X(:,6), X(:,7), X * theta, 'r')
hold off;

fprintf('Predictions for Specific Values:\n');

random = floor(rand(1) * m);
X_predict = X(random, :)
y_predict = y(random);
prediction = X_predict * theta;

fprintf('Prediction %f for real value (%f)\n', prediction, y_predict);

fprintf('\nEnd of program.\n');
