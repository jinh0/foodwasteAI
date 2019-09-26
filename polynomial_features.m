% Creates more features from the already existing variables
function [X_added_features] = polynomial_features(X, degrees)

n = size(X, 2);

% Create independent, separated polynomial variables
for d = 2:degrees
    X = [X, X .^ d];
end

% Intermingling of variables
for i = 1:degrees
    for j = i + 1:degrees
        X = [X, X(:, i) .* X(:, j)];
    end
end

X_added_features = X;

end
