# Normalize Features
function X_norm = normalize_features(X, mu, sigma)

m = size(X, 1);
X_norm = (X .- mu) ./ sigma;

end