% Creates more features from the already existing variables
function [X_added_features] = polynomial_features(X, degrees)

initial_n = size(X, 2);

X_add = [];

% Create independent, separated polynomial variables
% 3 loops for the 3 initial columns
for i = 0 : degrees % power for rice
    for j = 0 : degrees % power for Dal
        for k = 0 : degrees % power for students
            X_add = [X_add, (X(:, 1) .^ i) .* (X(:, 2) .^ j) .* (X(:, 3) .^ k)];
        end
    end
end

% Remove the column where all powers of variables are 0, which makes 1
X_add = X_add(:, 2 : size(X_add, 2));

% Merge initial X and X_add
X_added_features = [X, X_add];

end