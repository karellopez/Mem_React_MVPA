% function model = libsvm_multitarget_train(labels_train,data_train,cfg)
%
% Multitarget version for libsvm (training step)
%
% This function trains multiple models, one per label
%
% Labels are provided as cells (one per sample) with 1xn values (n: number 
% of models).
%
% Kai, 2020-12-10

function model = libsvm_multitarget_train(labels_train,data_train,cfg)

n_models = unique(cellfun(@length, labels_train));
assert(length(n_models)==1, 'libsvm_multitarget_train:different_number_of_labels', 'labels_train seems to have a different number of labels for different training samples, please check')
model = cell(1, n_models);

for model_ind = 1:n_models
    curr_labels_train = cellfun(@(x)x(model_ind), labels_train);
    model{model_ind} = libsvm_train(curr_labels_train,data_train,cfg); % get one model for each label index using the default function
end
