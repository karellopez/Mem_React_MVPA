% function decoding_out = libsvm_multitarget_test(labels_test,data_test,cfg,model)
%
% Multitarget version for libsvm (test step)
%
% This function trains multiple models, one per label
%
% Labels are provided as cells (one per sample) with 1xn values (n: number 
% of models).
%
% OUT
%   decoding_out.output{i}: decoding_out as from lisvm_test for model i
%
% See libsvm_test for more
% 
% Kai, 2020-12-10

function decoding_out = libsvm_multitarget_test(labels_test,data_test,cfg,model)

n_labels_per_sample = unique(cellfun(@length, labels_test));
assert(length(n_labels_per_sample)==1, 'libsvm_multitarget_test:different_number_of_labels', 'labels_test seems to have a different number of labels for different test samples, please check')
n_models = length(model);
assert(n_labels_per_sample==n_models, 'libsvm_multitarget_test:number_of_labels_does_not_match_number_of_models', 'the number of test labels per sample (%i) does not fit the number of models (%i). please check', n_labels_per_sample, n_models)
decoding_out = struct;

for model_ind = 1:n_models
    curr_labels_test = cellfun(@(x)x(model_ind), labels_test);
    curr_model = model{model_ind};
    
    % old name for model (model was stored but not with standard name)
    % decoding_out.output{model_ind} = libsvm_test(curr_labels_test,data_test,cfg,curr_model); % forward current model and respective labels to standard function
    decoding_out.model{model_ind} = libsvm_test(curr_labels_test,data_test,cfg,curr_model); % forward current model and respective labels to standard function
    decoding_out.opt{model_ind} = []; % not stored originally
end

% alternative way for e.g. one output per test (e.g. calculate euclidean
% distance)
% output = euclidean_distance(todo_calculate)


