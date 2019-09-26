function [X_added_features] = polynomial_features(X, degrees, num)

% indicates the original number of variables
initial_n = size(X, 2)

X_add_features = [];

% Independent polynomial variables

for d = 2:degrees
    X_add_features = [X_add_features, X(:, 1:initial_n) .^ d];
end

X = [X, X_add_features];

size(X, 2)

% Intermingling of variables
for i = 1:degrees
    for j = i+1:degrees
        X = [X, X(:,i) .* X(:,j)];
    end
    size(X, 2)
end

X_added_features = X(:,1:num);

end