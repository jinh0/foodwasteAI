% Normalize Features
% by altering X as z-score.
function X_norm = normalize_features(X, mu, sigma)

X_norm = (X .- mu) ./ sigma;

end
