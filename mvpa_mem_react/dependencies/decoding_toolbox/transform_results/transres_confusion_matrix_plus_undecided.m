% function output = transres_confusion_matrix_plus_undecided(decoding_out, chancelevel, varargin)
%
% Get a confusion matrix with an additional column if the class of a sample
% cannot be uniquely identified, e.g. if multiple classifiers produced a
% tie.
%
% This function has been written for libsvm, but problably works for other
% classifiers as well.
%
% Demo: demo3_1_simpletoydata_multiclass_confusion.m
%
% OUT
% The output will be an NxN+1 matrix where n is the number of unique labels.
% The N+1 column counts all undecided decisions. E.g. if a tie exist 
% between the class predicition for class 1 and 2 for a sample of class 1, 
% the value of (1, N+1) is increased.
%
% This code runs faster if all labels are in the same order in all decoding
% steps (e.g. runs).
%
% To use this transformation, use
%
%   cfg.results.output = {'confusion_matrix_plus_undecided'}
%
% Kai, 2021-11-16: Bugfix for more than 3 classes
%
% See also decoding_transform_results 
%   transres_accuracy_pairwise transres_confusion_matrix
%   transres_accuracy_pairwise_minus_chance transres_accuracy_matrix_minus_chance

% Hist
%   Kai, 2021-03-16: Created from transres_accuracy_matrix

function output = transres_confusion_matrix_plus_undecided(decoding_out, chancelevel, varargin)

% Code mainly copied from transres_accuracy_matrix

persistent keepind persistentlabels classes_per_classifier

n_step = length(decoding_out);
all_labels = vertcat(decoding_out.true_labels);
all_ulabels = uniqueq(all_labels);
n_label = size(all_ulabels,1);

%% Create index matrix and keep it during iterations (updated when labels change)
if ~isequal(persistentlabels, all_labels) % if this is the first iteration or there was any change
    persistentlabels = all_labels; % remember labels to avoid recomputation
    
    % we will run a check if all training labels in all runs are the same and have the same identity and count (not necessarily order) as the test labels
    test_labels = decoding_out(1).true_labels;
    if ~isequal(all_ulabels,uniqueq(test_labels))
        error('Some runs have different labels than others. transres_confusion_matrix_plus_undecided cannot deal with this case yet (sorry, it''s really difficult to code, see code for a detailed description). Please set up a design that does all pairwise classifications and use output sensitivity and specificity.')
    end
    prev = sort(decoding_out(1).model.Label);
    for i_step = 2:length(decoding_out)
        s = sort(decoding_out(i_step).model.Label);
        test_labels = decoding_out(i_step).true_labels;
        if (length(prev) ~= length(s)) || any(prev ~= s) || (n_label == length(s) && any(all_ulabels ~= s))
            error('Number and/or identity of training labels and/or test labels is not the same across all steps. Unfortunately transres_confusion_matrix_plus_undecided cannot deal with this case yet.  Please set up a design that does all pairwise classifications and use output sensitivity and specificity.')
        end
        if ~isequal(all_ulabels,uniqueq(test_labels))
            error('Some runs have different labels than others. transres_confusion_matrix_plus_undecided cannot deal with this case yet (sorry, it''s really difficult to code, see code for a detailed description). Please set up a design that does all pairwise classifications and use output sensitivity and specificity.')
        end
    end
    
    % For speeding everything up, we will use a matrix that defines for 
    % each column which class is predicted by a positive/negative dv.
    % E.g. in column 1, that classifies label1vslabel2, a positive value
    % predicts class 1.
    
    % we just need the position of 1, 2, etc. in the lower diagonal matrix,
    % with one row where each of them appears
    [x,y] = meshgrid(1:n_label);
    % pick lower diagonal
    keepind = tril(true(n_label),-1);
    classes_per_classifier = [x(keepind) y(keepind)]; 
    classes_per_classifier = classes_per_classifier';
    
    % these are the class labels for each column
    % e.g. ind(:,1) has the two class labels for the first classifier, the 
    % first is the class for dv>0, the second for dv<0. dv==0 is considered 
    % undecided and goes to the undecided class
end

%% Get pairwise accuracies from decision value matrix dv
% Note: expecting dv in canonical form, see e.g. tdt function libsvm_test.m
acc = zeros(n_label,n_label);
a = zeros(n_label,n_label);

% loop over all cross-validation iterations
% init output
confusion_table = zeros(n_label, n_label+1);

for i_step = 1:n_step
    
    test_labels = decoding_out(i_step).true_labels;
    
    % get the unique labels for the model
    % note: order of labels is not necessary anymore, because dv are 
    % ordered and brought to canonical form by libsvm_test.m
    modellabel = sort(decoding_out(i_step).model.Label); % Note: why sort(): libsvm values are not ordered, but decision values ARE now ordered by tdt function libsvm_test.m, so we can order the labels here

    % get the decision value matrix dv. 
    % dimension of dv: ntestsamples x npairwiseclassifiers.
    % dv contains in each column the decision value of a binary classifier, 
    % i.e. has as many columns as pairwise comparisions are possible.
    % E.g. for 3 classes with labels [1, 2, 4] three columns exist, for 
    % classifiers that classify 1vs2, 1vs4, 2vs4 (note: each classifiers
    % _test.m function should take care that the dv has this order, see
    % e.g. libsvm_test.m).
    % Each row then contains the decision value for each of these 
    % classifiers for each test sample, of course independent of the true 
    % label of the test sample.
    % Different options exist to get a class decision from these pairs,
    % which is what the dv matrix is normally used for. Here we use it
    % however to get a pairwise accuracy matrix from this.
   
    dv = decoding_out(i_step).decision_values; % dv: nsamples x n_binary_comparisons (1vs2, 1vs3, etc., see classes_per_classifier)
        
    % init class prediction with nan, undecided row comes later
    pred_class_mat = nan(size(dv));
    % get class predicted by each classifier and get majority vote 
    % plus undecided if no class gets a majority
    if isequal(all_ulabels(:),uniqueq(test_labels)) && size(dv, 2) == size(classes_per_classifier, 2) % each label occurs only once and in the correct order
        % dv now sorted by libsvm_test.m, not necessary to check label order anymore
        dv_plus = dv>0;
        dv_minus = dv<0;
        
        % get the majority
        plus_classes = repmat(classes_per_classifier(1, :), size(dv, 1), 1);
        pred_class_mat(dv_plus) = plus_classes(dv_plus);
        
        minus_classes = repmat(classes_per_classifier(2, :), size(dv, 1), 1);
        assert(all(isnan(pred_class_mat(dv_minus))), 'Something went wrong with assigning classes - no classes should be set for dv < 0 yet, but some are not nan. Check why.');
        pred_class_mat(dv_minus) = minus_classes(dv_minus);
        
        % get the majority for each row, ignoring nan
        [majority_pred_class, ~, other_modes] = mode(pred_class_mat, 2); % other_modes: if multiple classes have the same number of predictions, or if e.g. all are nan (one nan per class)
        % set ties to nan
        majority_pred_class(cellfun(@length, other_modes)~=1) = nan;
        
        % get indices of each predicted label (e.g. if negative labels
        % exist of or some do not exist) (I guess there are more elegent
        % versions than this, but this works)
        majority_pred_ind = repmat(1:size(all_ulabels,1), size(dv, 1), 1) == repmat(majority_pred_class, 1, size(all_ulabels,1));
        
        % assign all rows that have no entries to the undecided class
        undecided_filter = sum(majority_pred_ind, 2)~=1;
        majority_pred_ind(undecided_filter, end+1) = 1; %#ok<AGROW>
        
        % error if some have more than one
        assert(all(sum(majority_pred_ind, 2)==1), 'Unexpected error: At this point, each row should have exactly 1 entry 1, the rest 0');
                
        % assemble confusion table
        confusion_table_curr = nan(n_label, n_label+1);
        for true_label_ind = 1:length(all_ulabels)
            curr_true_label = all_ulabels(true_label_ind);
            curr_true_filter = decoding_out(i_step).true_labels == curr_true_label;
            n_curr_true = sum(curr_true_filter);
            for pred_label_ind = 1:size(majority_pred_ind, 2)
                curr_pred_filter = majority_pred_ind(:, pred_label_ind);
                curr_n = sum(curr_true_filter & curr_pred_filter);
                confusion_table_curr(true_label_ind, pred_label_ind) = curr_n; % raw values
                    % comment/uncomment the line before the final output to
                    % switch between raw values and row-normalised values
                    % (percent that add up to 100 for each true value)
            end
            
        end
        
    else % if not all training samples occur, maybe that is already covered above now? not checked
        error('Currently the function assumes that each cv fold has samples of all classes. Working with classifiers from cv folds that do not contain all classes is not implemented yet. Please check transres_accuracy_matrix on how to implement this here')
    end
    confusion_table = confusion_table + confusion_table_curr; % this is the confusion table for this run
end

% normalise to percent
% comment the following line if you want raw values, uncomment for percent
confusion_table = 100 * confusion_table ./ repmat(sum(confusion_table, 2), 1, size(confusion_table, 2));

output = {confusion_table};