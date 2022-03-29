clear;clc;close all;

data = load('alzheimers/ad_data.mat');
parameters = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
num_features = zeros(size(parameters,2),1);
aucs = zeros(size(parameters,2),1);

for i = 1:size(parameters, 2)
    % logistic l1 train
    parameter = parameters(i);
    [w, c] = logistic_l1_train(data.X_train, data.y_train, parameter);

    % get the number of features which are not equal to 0
    num_features(i) = sum(w ~= 0);

    % get the predictions and accuracy
    predictions = data.X_test * w + c;

    [~, ~, ~, auc] = perfcurve(data.y_test, predictions, 1);
    aucs(i) = auc;
end
%% draw the figure 
figure;
plot(parameters, num_features);
xlabel("L1 Regularization Parameter");
ylabel("Number of Features Selected");
title("Sparse Logistic Regression: NoF");

figure;
plot(parameters, aucs);
xlabel("L1 Regularization Parameter");
ylabel("AUC");
title("Sparse Logistic Regression: AUC");
