clear;clc;close all;

data = load('spam_email/data.txt');
labels = load('spam_email/labels.txt');

data = [data, ones(size(data,1),1)];
n = [200,500,800,1000,1500,2000];

test_data=data(2001:4601,:);
test_label=labels(2001:4601);

accuracy = zeros(size(n));

%% logistic train using different data size
for i = 1:length(n)
    train_data = data(1:n(i),:);
    train_label = labels(1:n(i));
    % logistic train
    weights = logistic_train(train_data,train_label);
    % compute prediction and get the accuracy
    prediction = 1./(1+exp(-(test_data*weights)));
    prediction(prediction >= 0.5) = 1;
    prediction(prediction < 0.5) = 0;
    accuracy(i) = mean(prediction == test_label);
end
% draw the figure
figure;
plot(n, accuracy, 'o-');
title('Logistic Regression: Experiment');
xlabel('Training data size');
ylabel('Testing Accuracy');