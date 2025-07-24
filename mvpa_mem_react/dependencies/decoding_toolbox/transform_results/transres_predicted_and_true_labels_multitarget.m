% function output = transres_predicted_and_true_labels_multitarget(decoding_out, chancelevel, varargin)
%
% Note: This functions is deprecated. It is equal (and just forwards to) 
% "transres_predicted_labels_multitarget", which also returns the true
% labels. If you prefer this name, you can use it, but it might lead to
% confusions, so we might remove it some time in the future.
%
% See 
% help transres_predicted_labels_multitarget
% (or header of the function) for more information.
%
% The only differences between both is how the output fields are called 
% (either "predicted_and_true_labels_multitarget" or just
% "predicted_labels_multitarget").
%
% To use this transformation, use
%
%   cfg.results.output = {'predicted_and_true_labels_multitarget'};
%
% NOTE: You can also use the shorter version 
%
%   cfg.results.output = {'predicted_labels_multitarget'};
%
% which does the same computation (and also returns the true labels), but
% uses a different name.
%
% Kai Görgen & Simon Weber, 2020-12-10, bugfix Kai 2025-02-19

function output = transres_predicted_and_true_labels_multitarget(decoding_out, chancelevel, varargin)

warning('TRANSRES_PREDICTED_AND_TRUE_LABELS_MULTITARGET:deprecated', 'transres_predicted_and_true_labels_multitarget() is deprecated because the name is too long and it does the same as another function.\nIt might be removed in future versions of the toolbox. Please use transres_predicted_labels_multitarget() instead.')
output = transres_predicted_labels_multitarget(decoding_out, chancelevel, varargin);