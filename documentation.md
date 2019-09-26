Functions Documentation
=======================

### **polynomial_features(X, degrees)** | Output: X_added_features

Creates more features in `X` to the number of degrees, `degrees`.

+ First, create new separated variables, using for loop on X .^ d. For example, create dal^3 and rice^2.
```
for d = 2:degrees
    X = [X, X .^ d];
end
```

+ Then, create new variables that cross multiply with each other.
```
for i = 1:degrees
    for j = i+1:degrees
        X = [X, X(:,i) .* X(:,j)];
    end
end
```

When `degrees = 3`, there are 7 variables.\
When `degrees = 4`, there are 30 variables.

### **splitdata(X, y)** | Output: X_training, X_test, y_training, y_test

Splits data into training and test data, executed on both X and y.

+ Training data: 80%, range = `1 : floor(0.8 * m)`
+ Test data: 20%, range = `floor(0.8 * m) : m`

### **generate_synthetic_data(mu, sigma, y_training, n)** | Output: X_fake, y_fake

TO DO

### **normalize_features(X, mu, sigma)** | Output: X_norm

Returns normalized values of X to prevent domination of a variable with bigger values.\
New values of X in `X_norm` (the z-score of X) is calculated by:
```
X_norm = (X .- mu) ./ sigma;
```
