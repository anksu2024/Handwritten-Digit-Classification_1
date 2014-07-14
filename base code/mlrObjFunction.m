function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size (D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ESTIMATION OF ERROR AND GRADIENT %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reshaping W vector to 716 X 10
W = reshape(W, size(X, 2) + 1, size(T, 2));

% x: Input with bias => 2000 X 716
x = [ones(size(X , 1) , 1) X];

% ank: Output Label based on the weights and input data => 2000 X 10
ank = x * W;

%%%%%sumCol = exp(ank);
%%%%%sum1 = sum(sumCol, 2);
%%%%%sum1 = repmat(sum1 , 1, 10);
sumCol = logsumexp(ank , 2);
sumCol = repmat(sumCol, 1 , size (ank , 2));

%%%%y = ank ./ sum1;

% logynk => 2000 X 10
logynk = ank - sumCol;

% error is a scalar
error = - sum(sum(T .* logynk));
%%%%error = (-1) .* sum(sum(T .* log(y)));
% error_grad: Gradience of Error => 716 X 10
error_grad = transpose(x) * (exp(logynk) - T);

%%%%error_grad = transpose(x) * (y - T);
error_grad = reshape(error_grad , size(error_grad, 1) * size(error_grad, 2) , 1);
end