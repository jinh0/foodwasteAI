function X_denorm = denormalize_features(X, mu, sigma)

X_denorm = X .* sigma .+ mu;

end
