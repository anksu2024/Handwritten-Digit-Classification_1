function [label] = mlrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of multi-class Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PREDICTION OF TRUE LABEL %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
label = zeros(size(X, 1), 1);

% x: Input with bias => 50000 X 716
x = [ones(size(X , 1) , 1) X];

% ank: Output Label based on the weights and input data => 50000 X 10
ank = x * W;

% logynk => 50000 X 10
%%%%%%logynk = zeros(size(ank , 1) , size (ank , 2));
sumCol = logsumexp(ank , 2);
sumCol = repmat(sumCol, 1 , size (ank , 2));

logynk = ank - sumCol;

for i = 1 : size(label , 1)
    label (i , 1) = find(exp(logynk(i , :)) == exp(max(logynk(i , :))) , 1);
end
end