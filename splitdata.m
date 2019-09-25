function [X_training, X_test, y_training, y_test] = splitdata(X, y)

m = size(X, 1);
n = size(X, 2);

X_training = X(1:floor(0.8 * m), :);
y_training = y(1:floor(0.8 * m));
X_test = X(floor(0.8 * m):m, :);
y_test = y(floor(0.8 * m):m, :);
