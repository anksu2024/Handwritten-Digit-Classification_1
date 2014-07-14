function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size ((D+1) X 10 , 1) represents the initial weight matrix 
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ESTIMATION OF ERROR AND GRADIENT %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x: Input with bias => 2000 X 716
x = [ones(size(X , 1) , 1) X];

N = size(x , 1);
D = size(x , 2);
classes = size(T , 2);
% initial_W = reshape(initital_W , N * classes , 1);

for i = 1 : n_iter
    ank = x * reshape(initial_W , D , classes);

    sumCol = logsumexp(ank , 2);
    
    % Replicating the columns to the number of classes available
    sumCol = repmat(sumCol, 1 , size (ank , 2));

    % logynk => 2000 X 10
    logynk = ank - sumCol;

    ynk = exp(logynk);

    % error_grad: Gradience of Error => 716 X 10
    error_grad = transpose(x) * (ynk - T);
    error_grad = reshape(error_grad , D * classes , 1);
    % We are good
    %%%%%%%%%%%%%%%
    
    I = eye(classes);
    
    % H => 7160 X 7160
    H = [];
    for k = 1 : classes
        blockH = [];
        for j = 1 : classes
            % h = zeros(D , D);
            % for n = 1 : N
            h = transpose(ynk(: , k)) * (I(k , j) - ynk(: , j)) * (transpose(x) * x);
            blockH = horzcat(blockH , h);
            % end
        end
        H = vertcat(H , blockH);
    end

    %initial_W = initial_W - pinv(H) * error_grad;
    initial_W = initial_W - pinv(H) * error_grad;
end

W = reshape(initial_W , D , classes);
end