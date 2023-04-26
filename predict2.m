% Student Prediction using rice and dal data v0.4
%   Split data + Synthetic Data Method 2
%   Regularization
%   Automated polynomial variables + output, Linear Regression Model

% Access path to functions
addpath('functions')

data = load('foodwaste2017/dinner/total_dinner.txt');

% Initialization of useful variables
X = data(:,1:3);
y = data(:,4);

fprintf('Student Prediction v0.4\n');
fprintf('Using rice and dal data + automated polynomial features,\nregularization, and synthetic + split data\n\n');

degrees = input('Degree of polynomial variables (default 1): ');

% Add polynomial variables
if degrees > 1
    X = polynomial_features(X, degrees);
end

% Split data
[X_training, X_test, y_training, y_test] = splitdata(X, y);

% More useful variables
m = size(X_training, 1);
n = size(X_training, 2)

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
X_training = [ones(size(X_training, 1), 1), X_training(:, 1:3)];

k=linspace(-10,10,50);
l=linspace(-10,10,50);

m = size(X_training, 1);
asdf = zeros(50, 50);
for i = 1:50
  for j = 1:50
    tmp = [1;  k(i); l(j); 1];
    asdf(i, j) = sum(((X_training * tmp) - y_training) .^ 2) / (2 * m);
  end
end

[xx,yy]=meshgrid(k,l);
mesh(xx,yy,asdf)

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('Gradient descent applied.\n\n');

% Gradient descent
theta = [1; rand(2, 1); 1];
alpha = 0.01;
iterations = 3000;

[theta, J_history, theta_history] = gradient_descent(X_training, y_training, theta, alpha, iterations);

theta_history(3000, :)
theta
plot(1:1500, J_history(1:1500));
xlabel('Number of Iterations');
ylabel('Cost');

fprintf('Program paused. Press enter to continue.\n');
pause;

k=linspace(-4,4,50);
l=linspace(-4,4,50);

m = size(X_training, 1);
asdf = zeros(50, 50);
for i = 1:50
  for j = 1:50
    tmp = [1;  k(i); l(j); 1];
    asdf(i, j) = sum(((X_training * tmp) - y_training) .^ 2) / (2 * m);
  end
end

minVal = min(min(asdf));
[minx, miny] = find(asdf == minVal);

plot3(theta_history(:, 2), theta_history(:, 1), J_history + 0.5, 'Linewidth', 3)
hold on;
plot3(l(miny), k(minx), minVal, '*', 'markersize', 5)
[xx,yy]=meshgrid(k,l);
mesh(xx,yy,asdf)
hold off;

%hold on;
%plot3([theta_history(1, 1), theta_history(1, 2)], [theta_history(2, 1), theta_history(2, 2)], [J_history(1), J_history(1 + 1)], '-')
%for i = 2:100
%  plot3([theta_history(i, 1), theta_history(i, 2)], [theta_history(i + 1, 1), theta_history(i + 1, 2)], [J_history(i), J_history(i + 1)], '-')
%end
%hold off;

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

lambda = 3;
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
title('Training Data against Predicted Points');

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
