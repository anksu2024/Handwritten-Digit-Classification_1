function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
% blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
% Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% WEIGHTS UPDATION USING NR METHOD%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X : Input Data => 50000 X 715 (Will have to add bias)
% x : Input Data with Bias => 50000 X 716
x = [ones(size(X , 1) , 1) X];

% x Represents Theta in Equations for finding w using NR Method

for i = 1 : n_iter
    % y => 50000 X 1
    y = sigmoid(x * initial_w);
    
    % R => 50000 X 50000
    R = y .* (1 - y);
    R = diag(sparse(R));
    
    % deltaEw: Gradient => 716 X 1
    deltaEw = transpose(x) * (y - t);
    
    % H: Hessian => 716 X 716
    H = transpose(x) * R * x;
    initial_w = initial_w - pinv(H) * deltaEw;
end

% w : Final weight Matrix => 716 X 1
w = initial_w;
end