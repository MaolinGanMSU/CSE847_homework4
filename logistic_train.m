function [weights] = logistic_train(data, labels, epsilon, maxiter)

if(~exist('epsilon','var'))
    epsilon = 1e-5;  % If the variable is not present, assign a value to it
end

if(~exist('maxiter','var'))
    maxiter = 1000;  % If the variable is not present, assign a value to it
end

data_length = size(data, 1);
weights = zeros(size(data, 2), 1);
iter = 1;
predict_cost = 1;
while iter < maxiter
    % get the new weight value
    old_cost = predict_cost;
    estimated_label = 1./(1+exp(-(data*weights)));
    diff_l = data'*(estimated_label-labels);
    diff_w = (1/data_length) * diff_l;
    weights = weights - diff_w;

    % get the absolute difference in prediction
    l = (1-labels).*(log(1-estimated_label));
    e = labels.*log(estimated_label);
    s = sum(e+l);
    predict_cost = (-1/data_length)*s;
    diff_predict = abs(predict_cost-old_cost);

    % update the condition of iteration
    if (diff_predict < epsilon)
        break;
    end
    iter = iter + 1;
end