% Creates more features from the already existing variables
%   USES number of variables desired, not degrees
function [X_added_features] = polynomial_features1(X, num)

initial_n = size(X, 2);

degrees = round((num - 2) ** (1/3));

X_add = [];

% Create independent, separated polynomial variables
% 3 loops for the 3 initial columns
for k = 0 : degrees % power for students
    for i = 0 : degrees % power for rice
        for j = 0 : degrees % power for dal
            X_add = [X_add, (X(:, 1) .^ i) .* (X(:, 2) .^ j) .* (X(:, 3) .^ k)];
        end
    end
end

% Remove the column where all powers of variables are 0, which makes 1
X_add = X_add(:, 2:(num - 2));

% Merge initial X and X_add
X_added_features = [X, X_add];

end
