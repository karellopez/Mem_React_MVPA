% function model = liblinear_train(labels_train,data_train,cfg)
%
% Wrapper function for liblinear train.
%
% This function trains a model using liblinear.
% The model can be applied to data with liblinear_test.
%
% Note that liblinear takes different parameters than libsvm.
% To check which, first type ">> which train" to check that the libsvm 
% train function is used.
% Then type "train" (without arguments) to check which parameters liblinear 
% takes.
% 
% Note especially that the bias term uses "-B" (not "-b" as libsvm).
%
% E.g., set
%   cfg.decoding.train.classification.model_parameters = '-s 1 -B 0 -q'
% for 
% -s 1: multi-class classification L2-regularized L2-loss support vector
%       classification (dual) (default), 
% -B: bias of 0 (default) (! -B, not -b, as for libsvm)
% -q: no output
%
% The performance of this function could be improved by passing a 
% precomputed kernel, but this has not be implemented yet (see
% libsvm_train if you want to implement it).
%
% Last adapted by Kai, 2024-10-11 (thanks Tobi!)

function model = liblinear_train(labels_train,data_train,cfg)

if isstruct(data_train), error('This method requires training vectors in data_train directly. Probably a kernel was passed method is use. This method does not support kernel methods'), end

switch lower(cfg.decoding.method)

    case 'classification'
        model = train(labels_train,sparse(data_train),cfg.decoding.train.classification.model_parameters);
        if isempty(model)
            error('libsvm_train:classification_model_empty', ...
                ['liblinear_train returned an empty model. ' ...
                'This is most likely due to wrong parameters. ' ...
                'Type ">> which train" to check that the libsvm train function is used. ' ...
                'Then type "train" (without arguments) to check which parameters liblinear takes. ' ...
                'Note especially that the bias term uses "-B" (not "-b" as libsvm). ' ...
                'The current parameters are: ' ...
                'cfg.decoding.train.classification.model_parameters = ' cfg.decoding.train.classification.model_parameters])
        end
        
    case 'classification_kernel'
        % Develop: If you implement this, adapt error at the beginning
        error('liblinear_train doesn''t work with passed kernels at the moment - please use libsvm or another method instead.')
        
    case 'regression'
        model = train(labels_train,sparse(data_train),cfg.decoding.train.regression.model_parameters);
        if isempty(model)
            error('libsvm_train:regression_model_empty', ...
                ['liblinear_train returned an empty model. ' ...
                'This is most likely due to wrong parameters. ' ...
                'Type ">> which train" to check that the libsvm train function is used. ' ...
                'Then type "train" (without arguments) to check which parameters liblinear takes. ' ...
                'Note especially that the bias term uses "-B" (not "-b" as libsvm). ' ...
                'The current parameters are: ' ...
                'cfg.decoding.train.regression.model_parameters = ' cfg.decoding.train.regression.model_parameters])
        end
        
    otherwise
        error('Unknown decoding method %s for cfg.decoding.software = %s',...
            cfg.decoding.method, cfg.decoding.software)
end