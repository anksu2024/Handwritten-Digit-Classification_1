%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% PROJECT 3 : HANDWRITTEN DIGITS CLASSIFICATION %%%%%%%%%%
%%%%% TEAM      : KARTHICK KRISHNA VENKATAKRISHNA   %%%%%%%%%%
%%%%%             RAHUL SINGH                       %%%%%%%%%%
%%%%%             ANKIT SARRAF                      %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOGISTIC REGRESSION WITH GRADIENT DESCENT %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

% PREPROCESS THE AVAILABLE DATASET
[train_data , train_label , validation_data , validation_label , ...
    test_data , test_label] = preprocess();

% SAVE THE RESULTS IN A NEW MAT FILE
save('dataset.mat' , 'train_data' , 'train_label' , ...
    'validation_data' , 'validation_label' , 'test_data' , 'test_label');
load('dataset.mat');

% n_class IS THE TOTAL NUMBER OF DISTINCT CLASSES
n_class = 10;

T = zeros(size(train_label, 1), n_class);

for i = 1 : n_class
    T(:, i) = (train_label == i);
end

tic;

% options => Number of Iterations to converge to correct value
options = optimset('MaxIter', 200);

% W is the matrix for the weights
W = zeros(size(train_data, 2) + 1, n_class);

initialWeights = zeros(size(train_data, 2) + 1, 1);

for i = 1 : n_class
    objFunction = @(params) blrObjFunction(params , train_data , T(:, i));
    [w, ~] = fmincg(objFunction , initialWeights , options);
    W(:, i) = w;
end

predicted_label1 = blrPredict(W, train_data);
fprintf('\nTraining Set Accuracy: %f\n', ...
                mean(double(predicted_label1 == train_label)) * 100);

predicted_label2 = blrPredict(W, validation_data);
fprintf('\nValidation Set Accuracy: %f\n', ...
                mean(double(predicted_label2 == validation_label)) * 100);

predicted_label3 = blrPredict(W, test_data);
fprintf('\nTest Set Accuracy: %f\n', ...
                mean(double(predicted_label3 == test_label)) * 100);

fprintf('\nTotal Processing Time: %f seconds\n' , toc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOGISTIC REGRESSION WITH NEWTON-RAPHSON METHOD %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

% PREPROCESS THE AVAILABLE DATASET
[train_data , train_label , validation_data , validation_label , ...
    test_data , test_label] = preprocess();

% SAVE THE RESULTS IN A NEW MAT FILE
save('dataset.mat' , 'train_data' , 'train_label' , ...
    'validation_data' , 'validation_label' , 'test_data' , 'test_label');
load('dataset.mat');

% n_class IS THE TOTAL NUMBER OF DISTINCT CLASSES
n_class = 10;

T = zeros(size(train_label, 1), n_class);

for i = 1 : n_class
    T(:, i) = (train_label == i);
end

tic;

% Weight Matrix which would be found using train_data
W = zeros(size(train_data , 2) + 1 , n_class);

% initialized Weights
initialWeights = zeros(size(train_data, 2) + 1 , 1);

% Iterations while finding the Weights in NR Method
% Note the readings for the parameter n_iter = 5, 10, 15
n_iter = 5;

for i = 1 : n_class
    W(:, i) = blrNewtonRaphsonLearn(initialWeights , ...
        train_data, T(:, i), n_iter);
end

predicted_label1 = blrPredict(W , train_data);
fprintf('\nTraining Set Accuracy: %f\n' , ...
    mean(double(predicted_label1 == train_label)) * 100);

predicted_label2 = blrPredict(W , validation_data);
fprintf('\nValidation Set Accuracy: %f\n' , ...
    mean(double(predicted_label2 == validation_label)) * 100);

predicted_label3 = blrPredict(W, test_data);
fprintf('\nTest Set Accuracy: %f\n' , ...
    mean(double(predicted_label3 == test_label)) * 100);

fprintf('\nTotal Processing Time: %f seconds\n' , toc);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MULTICLASS LOGISTIC REGRESSION WITH GRADIENT DESCENT %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

% Load the new Dataset for MLR
load('newdataset_MLR.mat');

% n_class IS THE TOTAL NUMBER OF DISTINCT CLASSES
n_class = 10;

T = zeros(size(train_label, 1), n_class);

for i = 1 : n_class
    T(:, i) = (train_label == i);
end

tic;

options = optimset('MaxIter' , 200);
initialWeights = zeros((size(train_data , 2) + 1) * n_class , 1);

objFunction = @(params) mlrObjFunction(params, train_data , T);
[W , cost] = fmincg(objFunction , initialWeights , options);
W = reshape(W , size(train_data , 2) + 1 , n_class);

predicted_label1 = mlrPredict(W , train_data);
fprintf('\nTraining Set Accuracy: %f\n' , ...
        mean(double(predicted_label1 == train_label)) * 100);

predicted_label2 = mlrPredict(W , validation_data);
fprintf('\nValidation Set Accuracy: %f\n' , ...
        mean(double(predicted_label2 == validation_label)) * 100);

predicted_label3 = mlrPredict(W , test_data);
fprintf('\nTest Set Accuracy: %f\n' , ...
        mean(double(predicted_label3 == test_label)) * 100);

fprintf('\nTotal Processing Time: %f seconds\n' , toc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MULTICLASS LOGISTIC REGRESSION WITH NEWTON-RAPHSON METHOD %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

% Load the new Dataset for MLR
load('newdataset_MLR.mat');

% n_class IS THE TOTAL NUMBER OF DISTINCT CLASSES
n_class = 10;

T = zeros(size(train_label, 1), n_class);

for i = 1 : n_class
    T(:, i) = (train_label == i);
end

tic;

initialWeights = zeros((size(train_data, 2) + 1) * n_class, 1);
n_iter = 5;
[W] = mlrNewtonRaphsonLearn(initialWeights, train_data, T, n_iter);

predicted_label = mlrPredict(W , train_data);
fprintf('\nTraining Set Accuracy: %f\n' , ...
    mean(double(predicted_label == train_label)) * 100);

predicted_label = mlrPredict(W , validation_data);
fprintf('\nValidation Set Accuracy: %f\n' , ...
    mean(double(predicted_label == validation_label)) * 100);

predicted_label = mlrPredict(W , test_data);
fprintf('\nTest Set Accuracy: %f\n' , ...
    mean(double(predicted_label == test_label)) * 100);

fprintf('\nTotal Processing Time: %f seconds\n' , toc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SUPPORT VECTOR MACHINE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;

load('newdataset_SVM.mat');

% things needed to perforn the SVM operations

cost_matrix = [1 10:10:100];


%for linear kernel with all the default parameters
model_linear = svmtrain ( train_label, train_data, ' -t 0');
[predicted_label_linear_train, accuracy_linear_train, ~] = svmpredict (train_label, train_data, model_linear);
[predicted_label_linear_validation, accuracy_linear_validation, ~] = svmpredict(validation_label, validation_data, model_linear);
[predicted_label_linear_test, accuracy_linear_test, ~] = svmpredict(test_label, test_data, model_linear);

%for radial basis function kernel with parameter gamma = 1
model_rbf_1 = svmtrain(train_label ,train_data ,' -t 2  -g 1');
[predicted_label_rbf1_train, accuracy_rbf1_train, ~] = svmpredict(train_label , train_data , model_rbf_1 ) ;
[predicted_label_rbf1_validation, accuracy_rbf1_validation, ~] = svmpredict(validation_label, validation_data, model_rbf_1);
[predicted_label_rbf1_test, accuracy_rbf1_test, ~] = svmpredict(test_label, test_data, model_rbf_1);

%for radial basis function kernel with default parameters
model_rbf_default = svmtrain(train_label, train_data, ' -t 2 ');
[predicted_label_rbf2_train, accuracy_rbf2_train, ~] = svmpredict(train_label, train_data, model_rbf_default);
[predicted_label_rbf2_validation, accuracy_rbf2_validation, ~] = svmpredict(validation_label, validation_data, model_rbf_default);
[predicted_label_rbf2_test, accuracy_rbf2_test, ~] = svmpredict(test_label, test_data, model_rbf_default);

%for radial basis function kerenel with different values of cost

accuracy_rbf3_train = zeros(length(cost_matrix), 1);
accuracy_rbf3_valid = zeros(length(cost_matrix), 1);
accuracy_rbf3_test = zeros(length(cost_matrix), 1);
max_accuracy = 0;
index = 0;

for i = 1 : length(cost_matrix)
    cost_param = sprintf('-t 2  -c %d', cost_matrix(i));
    model = svmtrain(train_label, train_data, cost_param);
    [predicted_label_rbf2_train, accuracy, ~] = svmpredict(train_label, train_data, model);
    accuracy_rbf3_train(i, 1) = accuracy(1);
    [predicted_label_rbf2_validation, accuracy, ~] = svmpredict(validation_label, validation_data, model);
    accuracy_rbf3_valid(i, 1) = accuracy(1);
    [predicted_label_rbf2_test, accuracy, ~] = svmpredict(test_label, test_data, model);
    accuracy_rbf3_test(i, 1) = accuracy(1);
    if(accuracy_rbf3_valid(i, 1) > max_accuracy)
        max_accuracy = accuracy_rbf3_valid(i, 1);
        max_index = i;
        model_rbf_c = model;
        cost_rbf_c = cost_matrix(i);
    end
end

save('params.mat', 'model_linear', 'model_rbf_1', 'model_rbf_default', 'model_rbf_c', 'cost_rbf_c');
