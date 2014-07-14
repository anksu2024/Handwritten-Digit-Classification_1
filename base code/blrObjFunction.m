function [error, error_grad] = blrObjFunction(w, X, t)
% blrObjFunction computes 2-class Logistic Regression error function and
% its gradient.
%
% Input:
% w: the weight vector of size (D + 1) x 1 
% X: the data matrix of size N x D
% t: the label vector of size N x 1 where each entry can be either 0 or 1
%    representing the label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size (D+1) x 1 representing the gradient of
%             error function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ESTIMATION OF ERROR AND GRADIENT %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DIMENSIONS (WHEN THE X = training_data)
% X => 50000 X 715
% w => 716 X 1
% t => 50000 X 1
% x is the input variable of dimensions => 50000 X 716
x = [ones(size(X , 1) , 1) X];
%x = ones(size(X , 1), size(w , 1));
%x(: , 1 : size(X , 2)) = X;

% y => 50000 X 1
y = sigmoid(x * w);

% error => scalar
error = -sum(t .* log(y) + (1 - t) .* log(1 - y));

% error_grad => 716 X 1
error_grad = transpose(x) * (y - t);
end