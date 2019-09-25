function [X_fake, y_fake] = generate_synthetic_data(mu, sigma, y, n)

random = randn(100, 1);
random = repmat(random, 1, n);
X_fake = (random .* sigma) .+ mu;
y_fake = random(:,1) .* std(y) + mean(y);

end