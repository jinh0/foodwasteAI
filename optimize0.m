%%% Optimization of Food Waste v0.1
%%% TO DO:
%%%     + Use different methods to minimize the waste
%%%     + Should return X from predicted y and theta

% Load data and separate into X and y
data = load('foodwaste2017/dinner/total_dinner.txt');
X = data(:,1:3);
y = data(:,4); 

% 