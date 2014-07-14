function [label] = blrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10. Each column is the weight
%    vector of a Logistic Regression classifier.
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PREDICTION OF TRUE LABEL %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DIMENSIONS (CONSIDER training_data FOR INSTANCE)
% X => 50000 X 715
% W => 716 X 10
% x => 50000 X 716
x = [ones(size(X , 1) , 1) X];

% y => 50000 X 10
y = sigmoid(x * W);

% label => 50000 X 1
label = zeros(size(X , 1) , 1);

% Store the row wise maximum value of y in the label
for i = 1 : size(label , 1)
    label (i , 1) = find(y(i , :) == max(y(i , :)) , 1);
end
end