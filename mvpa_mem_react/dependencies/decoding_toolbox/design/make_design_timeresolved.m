% function cfg.design = make_design_timeresolved(cfg)
%
% Function that takes in cfg settings and all possible conditions and
% timing information for constructing a cfg design. Returns a cfg with all
% design paramaters. Based on TDT functions.
%
% Currently, make_design_cv is used to create one cross-timepoint cv design 
% for each potential comparison, i.e. each combination of labels. 
% As conditions in classification are symmetric 
%   (accuracy(a vs b) == accuracy(b vs a)) only one comparison is done
%   for each combinations.
% As cross-classification (xclass) between timepoints is not symmetric
%   (different to train on timepoint 1 and test on timepoint 2 than vice
%   versa) each combination of timepoints is used for training and testing,
%   including training and testing on the same timepoint (equals standard 
%   cv without xclass).
% Additional information is provided that allow creating time-resolved 
%   accuracy matrices, one for each comparison.
%
% EXAMPLE use of output to create result matrices (as many as comparisons)
%   [comp_ind, train_time_ind, test_time_ind] = cfg.design.set2conditions_map(set, :);
%   result_matrix{comp_ind}(train_time_ind, test_time_ind) = results.accuracy.set(set_ind).output;
% where results is returned from decoding(cfg).
% The label vectors in cfg.design.conditions2set_map_axislabel{comp_ind} 
% can then be used to label the resulting matrices, e.g. to plot it or
% create a table with row and col names.
%
% IN
%  cfg: settings needed to construct design
%     required
%       cfg.files.timepoint: 1xn_files vector which timepoint for each file
%       cfg.files.chunk:     chunk numbers
%       cfg.files.name:      name of each file
%       cfg.files.label:     1xn_files vector which labelfor each file
%           currently all combinations of different unique labels are
%           classified as comparisons
%       cfg.design.make_pairwise: 1 if sets are tested pairwise or across
%           all classes
%       
%     optional
%       cfg.files.timepointstr: cellstr matching 1:1 timepoint, i.e. having
%           the same string for each cfg.files.timepoint
%       cfg.files.descr: cellstr with description of file, currently used
%           to extract labels for condition 
%        (! currently condition-specific text at same position in string!)
%       cfg.design.make_set2condition_map_str: if 1, a string
%           representation for set2condition_map_str is created. Leave out
%           or disable to save some time and space.
%       cfg.design.template_func: (experimental) function to use to create 
%           the template design, other than make_design_cv (e.g. 
%           @make_design_xclass_cv)
%               
% OUT
%
%  in addition to the design:
%    cfg.design.(set/train/test/label): required fields for decoding(cfg)
%
%  the following cfg fields are created:
%
%   for each COMPARISON (i.e. class a vs b) [currently extracted from unique(cfg.files.label), above]:
%    cfg.design.conditions2set_map{comparison_ind}(train_time_ind, test_time_ind)
%       struct with n_comparison matrices of size n_train_time, n_test_time
%       containing the set number that corresponds each coordinate 
%       (train_time x test_time). Can be used to put in a result matrix
%       (see explanation above). For x/y label, see below. 
%       Inverse: cfg.design.set2conditions_map
%    cfg.design.conditions2set_map_axislabel{comparison_ind}: x/ylabel for e.g.
%       creating a table or plot for the data in 
%       cfg.conditions2set_map{comparison_ind} or 
%       cfg.conditions2set_map_str{comparison_ind}. 
%       Contain subfields
%       traintime_yaxis_vals: [n_traintime × 1 double]
%       testtime_xaxis_vals: [1 x n_testtime double]
%       traintime_yaxis_str: {n_traintime × 1 cellstr}
%       testtime_xaxis_str: {1 × n_testtime cellstr}
%    
%   following fields for each SET
%    cfg.design.set2conditions_map: [n_set x 3 double]. Inverse of 
%       cfg.design.conditions_map2set map. Example use (for accuracy):
%         [comp_ind, train_time_ind, test_time_ind] =
%            cfg.design.set2conditions_map(set, :);
%         result_matrix{comp_ind}(train_time_ind, test_time_ind) =
%             results.accuracy.set(set_ind).output;
%    cfg.design.set2condition_map_str(full_set_ind):
%       cellstr with same elements as above, with text describing each entry
%       e.g. {'Comparison: 1, Train: time1, Test: time2'} for classifying
%       condition 1 vs 2 using training data from time1 and test data from
%       time2.
%    cfg.design.label_val2str_map: table mapping label to string 
%       representation. Only exist cfg.files.timepointstr contains strings 
%       that much 1:1 to cfg.files.timepoint
%    cfg.design.timepoint_val2str_map: table mapping timepoints to string
%       represetation. Only present if cfg.files.descr exists and if it has
%       a common part of text (! at the same position !)
%       Maybe TODO: Could be provided in a smarter way
%
%   following fields for each (CV) STEP
%    cfg.design.condition_list_index_per_step = zeros(1, n_total_steps);
%    cfg.design.conditions_per_step = zeros(size(condition_list, 2), n_total_steps);
%    cfg.design.train_timepoint_per_step = zeros(1, n_total_steps);
%    cfg.design.test_timepoint_per_step = zeros(1, n_total_steps);
%
%   and also
%    cfg.design.template_cfg: template used to create the design
%    cfg.design.fighandles.template_design: figure with template
%
%  version as
%    cfg.design.timeresolved_design.ver = ver;
%  This can be validated against the call below to e.g. validate caching.
%   
% Alternative use to validate cache
%   ver = timeresolved_design('ver');
% will return the version ver defined in the file
%   as stored in cfg.design.timeresolved_design.ver
%
% Kai and Ingmar 2020-05-26

% Fixed missing set2conditions_map_str. Does not make this on default. 
%   Call cfg.design.make_set2conditions_map_str = 1 
%   to get it. Ingmar & Kai 21-07-13
% Maybe todo: consider adding option to provide condition_list and the maps 
%   that are used to create text representations directly

function design = make_design_timeresolved(cfg)
%% return version information if wanted (to check cache validity only)

ver = 'v2021-07-13';
if nargin == 1 && ischar(cfg) && strcmp(cfg, 'ver')
    design = ver;
    return
end
cfg.design.timeresolved_design.ver = ver;

%% Unpack some cfg variables
% Note: change explanation in header if list or blocktimes are provided
% differently

if ~isfield(cfg.design, 'make_pairwise')
    cfg.design.make_pairwise = 0;
end

u_labels = unique(cfg.files.label);
if cfg.design.make_pairwise
    [a, b]  = meshgrid(u_labels); % all label vs all label
    condition_list = [a(:), b(:)]; % remove equal conditions
    
    % only keep each unique combination of labels once (training and 
    % testing on different time steps implemented later)
    condition_list(condition_list(:, 1) >= condition_list(:, 2), :) = [];

else
    condition_list(1, :) = u_labels;
end   

% currently: same times for training and testing
block_times     = unique(cfg.files.timepoint);
% check each use of block_times when you want to change this

%% verify that time points have 1:1 value and create map
% map unique time points and conditions strings to respective numbers

if isfield(cfg.files, 'timepointstr')
    timepoints_str = {}; timepoints = [];% names only used to create table row name
    for b_ind = 1:length(block_times)
        curr_uniques = unique(cfg.files.timepointstr(cfg.files.timepoint==block_times(b_ind)));
        if length(curr_uniques) > 1
            warning('make_design_timeresolved:multiple_timepointstr', ...
                'cfg.files.timepointstr does not uniquely correspond to cfg.files.timepoint==%i, taking all unique entires as description', block_times(b_ind));
        elseif isempty(curr_uniques)
            warning('make_design_timeresolved:empty_timepointstr', ...
                'cfg.files.timepointstr is empty for cfg.files.timepoint==%i', block_times(b_ind));
        else
            % all fine, nothing to do
        end
        % add entry
        timepoints(end+1, 1) = block_times(b_ind); %#ok<AGROW>
        timepoints_str(end+1, 1) = {curr_uniques}; %#ok<AGROW>
    end
    
    % check that the list of strings maped contains all values (i.e. no double)
    if length(unique([timepoints_str{:}])) ~= length(unique(timepoints))
        warning('make_design_timeresolved:differnt_unique_length', ...
            'cfg.files.timepoints_str and cfg.files.timepoints have a different number of unique elements')
    end
    
    % store as map (table)
    t = table(timepoints, timepoints_str);
    cfg.design.timepoint_val2str_map = t;
    clear timepoints timepointsstr % names only used to create table row name
end

%% try to extract a condition from file.descr that have 1:1 value and create map
if isfield(cfg.files, 'descr')    
    common_descr = {}; label = [];% names only used to create table row name
    % find all characters that are equal for all descriptions
    curr_descrs = cfg.files.descr;
    curr_descrs = char(curr_descrs(:)); % convert to char and force vector
    ds = diff(curr_descrs); % differences of asci value
    % all columns that have no difference
    all_equal_idx = sum(abs(ds))==0;
    
    for l_ind = 1:length(u_labels)
        % we assume the name is equal at the same position
        curr_descrs = cfg.files.descr(cfg.files.label==u_labels(l_ind));
        curr_descrs = char(curr_descrs(:)); % convert to char and force vector
        ds = diff(curr_descrs); % differences of asci value
        % all columns that have no difference
        curr_all_equal_idx = sum(abs(ds))==0;
        label(l_ind, 1) = u_labels(l_ind); %#ok<AGROW>
%         % alterantive one: keep only what current have in common and mask
%         % rest with ...
%         common_descr{l_ind, 1} = repmat('.', 1, size(curr_descrs, 2)); %#ok<AGROW>
%         common_descr{l_ind, 1}(curr_all_equal_idx) = curr_descrs(1, curr_all_equal_idx); %#ok<AGROW> % keep what current, but not ALL have in common
        % keep only what current but not all have in common
        common_descr{l_ind, 1} = curr_descrs(1, curr_all_equal_idx & ~all_equal_idx); %#ok<AGROW> % keep what current, but not ALL have in common
    end

    % store as map (table)
    t = table(label, common_descr);
    cfg.design.label_val2str_map = t;
    clear common_descr label % names only used to create table row name
end

%% Get template for next step ignoring time resolution and xclass so here
% Getting desired design (e.g. cv) design for all conditions at once 
% This will be expanded for time resolution and/or xclass in the next step

template_cfg = cfg;
if ~isfield(cfg.design, 'template_func')
    template_cfg.design = make_design_cv(template_cfg);
else
    % e.g. xclass cv design: w()
    template_cfg.design = feval(cfg.design.template_func, template_cfg); 
end 
cfg.design.template_cfg = template_cfg; % save to cfg

%plot only for 1 or 2 subjects, not while looping
cfg.design.fighandles.template_design = plot_design(template_cfg);
title('Template design');
set(cfg.design.fighandles.template_design, 'name', 'template_design');

%% Expand to get all combinations

n_comparisons = size(condition_list, 1); % number of comparisons, e.g. Aeh vs Kon, Kon vs Sum, Sum vs. Aeh
n_timepoints = numel(block_times); % number of different time points to compare

% Outcome are n_comparison matrices (size traintimes x testtimes)
% NOT one big matrix (ncompxtraintimes) x (ncomp*testtimes).
% Reason: It does not make sense to change labels between training and test set
n_cv_steps_per_analysis = numel(template_cfg.design.set); % number of cv steps
n_combinations = n_comparisons * n_timepoints^2; 
n_total_steps = n_combinations * n_cv_steps_per_analysis;

dispv(1, 'total_steps: %i (= %i comparisons x [%i timepoints]^2 x %i cv_steps_per_analysis)\n', ...
    n_total_steps, n_comparisons, n_timepoints, n_cv_steps_per_analysis)
dispv(1, 'Allocating space for %i total_steps x %i files\n', n_total_steps, numel(cfg.files.name));

% Prepare .design
% copy all fields from template to cfg
field_names = fieldnames(template_cfg.design);
for f_ind = 1:length(field_names)
    assert(~isfield(cfg.design, field_names{f_ind}), ...
        'Cannot copy field from template_cfg: Target field cfg.design.%s already exist. Please check why.', field_names{f_ind});
    cfg.design.(field_names{f_ind}) = template_cfg.design.(field_names{f_ind});
end
cfg.design = rmfield(cfg.design, 'function'); % remove function, from here on custom code

% assert so far the number of rows is correct
assert(numel(cfg.files.name) == size(cfg.design.train, 1) && size(cfg.design.train, 1) == size(cfg.design.test, 1), ...
    'Wrong dimensions in cfg.design. .files, .train, .test need to have the same number of rows (size(.., 1)). Please check why.');

% Preallocate set, train, test
cfg.design.set   = zeros(1, n_total_steps);
cfg.design.train = zeros(size(cfg.design.train, 1), n_total_steps);
cfg.design.test  = zeros(size(cfg.design.test, 1),  n_total_steps);
cfg.design.label = repmat(cfg.design.label, 1, n_combinations); % labels never change for different analyses, so calculate directly

% add description for each step
cfg.design.condition_list_index_per_step = zeros(1, n_total_steps);
cfg.design.conditions_per_step = zeros(size(condition_list, 2), n_total_steps);
cfg.design.train_timepoint_per_step = zeros(1, n_total_steps);
cfg.design.test_timepoint_per_step = zeros(1, n_total_steps);

% allocate set descriptions (inverse condition2map initialised below)
% init: n_sets (n_combinations) x (comparison_ind (matrix) x traintime(yaxis) x testtime(xaxis)) for each set
cfg.design.set2conditions_map = nan(n_combinations, 3); 

% and init strmap if used
if isfield(cfg.design, 'make_set2conditions_map_str') && cfg.design.make_set2conditions_map_str
    cfg.design.set2conditions_map_str = cell(n_combinations, 1); % init
end

dispv(1, '    ... Allocation successful.\n')

%% loop all design combinations

dispv(1, 'Constructing full cfg for decoding: Creating all combinations')

% asserts
if ~isequal(template_cfg.files.name, cfg.files.name)
    error('~isequal(template_cfg.files.name, cfg.files.name), no idea why. They are assumed to be equal below. Make sure they are.')
end

% init
full_cfg_cvstep = 0; % init counter for current column
full_cfg_set    = 0; % init counter for set
start_time      = now;
msg_length      = 0;

% loops
for comparison_ind = 1:n_comparisons
    % get names of current conditions
    curr_cond = condition_list(comparison_ind, :);
    
    % init matrix placing set numbers in train-test times matrix for this
    % comparison
    cfg.design.conditions2set_map{comparison_ind} = nan(numel(block_times), numel(block_times)); % init: train x test time matrix (currently square)
    cfg.design.conditions2set_map_axislabel{comparison_ind}.traintime_yaxis_vals(:, 1) = block_times; % put train times as row vector (ylabel)
    cfg.design.conditions2set_map_axislabel{comparison_ind}.testtime_xaxis_vals(1, :) = block_times; % put test times as col vector (xlabel)
    % inverse mapping set2conditions_map initialised above

    % create train and test time string labels 
    train_vals = cfg.design.conditions2set_map_axislabel{comparison_ind}.traintime_yaxis_vals;
    test_vals  = cfg.design.conditions2set_map_axislabel{comparison_ind}.testtime_xaxis_vals;

    % NOTE: double code for generating timepoint_str: look for isfield(cfg.design,  'timepoint_val2str_map') 
    % and change code accordingly if you change the code here
    if isfield(cfg.design,  'timepoint_val2str_map') % use string representation if present
        train_strs = cellfun(@(x)[cfg.design.timepoint_val2str_map.timepoints_str{cfg.design.timepoint_val2str_map.timepoints==x}{:}], ...
            num2cell(train_vals), 'UniformOutput', false);
        test_strs  = cellfun(@(x)[cfg.design.timepoint_val2str_map.timepoints_str{cfg.design.timepoint_val2str_map.timepoints==x}{:}], ...
            num2cell(test_vals), 'UniformOutput', false);
    else % use generated text 
        train_strs = cellfun(@(x)@(x)sprintf('time%i', x), num2cell(train_vals), 'UniformOutput', false);
        test_strs  = cellfun(@(x)@(x)sprintf('time%i', x), num2cell(test_vals),  'UniformOutput', false);
    end
    cfg.design.conditions2set_map_axislabel{comparison_ind}.traintime_yaxis_str = train_strs;
    cfg.design.conditions2set_map_axislabel{comparison_ind}.testtime_xaxis_str  = test_strs;

    % loop over training times
    for train_time_ind = 1:numel(block_times)
        
        % display progress (change indexing if you move this somewhere else
        if mod(train_time_ind, 1) == 0 % option: every n-th step
            n_total = n_comparisons * numel(block_times);
            
            % add test index if in test loop
            cnt = (comparison_ind-1)*numel(block_times) + train_time_ind; 
            try
                % nice version from TDT
                dp_cfg.analysis = 'Expanding design';
                msg_length = display_progress(dp_cfg,cnt,n_total,start_time,msg_length);
            catch
                % quick version that also works
                dispv(1, '  Processing: cond %i/%i, train %i/%i(%g%%)\n', ...
                    comparison_ind, n_comparisons, train_time_ind, numel(block_times), 100*cnt/n_total)
            end
        end
        
        % get phase for training
        train_time     = block_times(train_time_ind);
        
        % init cache for training indices
        curr_cond_filter_train = cell(n_cv_steps_per_analysis, length(curr_cond));
        
        % get data for training (caching cv_ind for speedup)
        for cv_ind = 1:n_cv_steps_per_analysis
            for cond_ind = 1:length(curr_cond)
                % find rows to copy
                curr_cond_filter_train{cv_ind, cond_ind} = ...
                    template_cfg.files.label == curr_cond(cond_ind) & ...
                    template_cfg.files.timepoint == train_time;

                if sum(curr_cond_filter_train{cv_ind, cond_ind}) == 0
                    error('Unexpected: Could not find condition in template, please check')
                end
                % get from destination; note: data has been put in target in loop over test times
                curr_train_dat{cv_ind, cond_ind} = template_cfg.design.train(curr_cond_filter_train{cv_ind, cond_ind}, cv_ind); %#ok<AGROW> 
            end
        end
        
        % loop over test times
        for test_time_ind = 1:numel(block_times)
            
            % get phase for training
            test_time = block_times(test_time_ind);
            
            % increase SET counter
            full_cfg_set = full_cfg_set + 1; % SET, not step
            
            % add set description ("coordinate" in train test matrix)
            % list style (set) -> comparison_ind, train_time_ind, test_time_ind
            assert(all(isnan(cfg.design.set2conditions_map(full_cfg_set, :))), ...
                'Condition comparison %i, train %i, test %i: cannot assign set in comparison/y/x table at position %i. Position already contains values %i, %i, %i', ...
                full_cfg_set, comparison_ind, train_time_ind, test_time_ind, cfg.design.set2conditions_map(full_cfg_set, :));
            cfg.design.set2conditions_map(full_cfg_set, :) = [comparison_ind, train_time_ind, test_time_ind];
            
            % create set2conditions_map_str (pre-allocated earlier)
            if isfield(cfg.design, 'make_set2conditions_map_str') && cfg.design.make_set2conditions_map_str
                cfg.design.set2conditions_map_str{full_cfg_set,:} = ['Comparison: ' num2str(comparison_ind) ', Train: ' train_strs{train_time_ind} ', Test: ' test_strs{test_time_ind}];
            end
            
            % table style 
            % ensure its not taken {comparison_ind}(train_time_ind, test_time_ind)
            curr_val = cfg.design.conditions2set_map{comparison_ind}(train_time_ind, test_time_ind);
            assert(isnan(curr_val), 'Condition comparison %i, train %i, test %i: cannot assign set %i. Position already taken by set %i', full_cfg_set, curr_val);
            cfg.design.conditions2set_map{comparison_ind}(train_time_ind, test_time_ind) = full_cfg_set; 
            
            % get and set entries for .train, .test, .set per STEP
            for cv_ind = 1:n_cv_steps_per_analysis
                
                % increase column counter in full cfg
                full_cfg_cvstep = full_cfg_cvstep + 1; % STEP, not set
                
                % set current set value in full cfg
                cfg.design.set(1, full_cfg_cvstep) = full_cfg_set;
                
                % copy train conditions from template to full design matrix
                for cond_ind = 1:numel(curr_cond)
                    % put train in target
                    cfg.design.train(curr_cond_filter_train{cv_ind, cond_ind}, full_cfg_cvstep) = curr_train_dat{cv_ind, cond_ind};
                end
                
                % copy test conditions from template to full design matrix
                for cond_ind = 1:numel(curr_cond)

                    curr_cond_filter = template_cfg.files.label == curr_cond(cond_ind) & ...
                        template_cfg.files.timepoint == test_time;
                        
                    if sum(curr_cond_filter) == 0
                        error('Unexpected: Could not find condition in template, please check')
                    end
                    % get from destination
                    curr_dat = template_cfg.design.test(curr_cond_filter, cv_ind);
                    % put in target
                    cfg.design.test(curr_cond_filter, full_cfg_cvstep) = curr_dat;
                    
                    % add description of current conditions 
                    cfg.design.condition_list_index_per_step(1, full_cfg_cvstep)    = comparison_ind; % row of conditions_list that contains current conditions
                    cfg.design.conditions_per_step(:, full_cfg_cvstep)              = curr_cond; % current conditions
                    cfg.design.train_timepoint_per_step(1, full_cfg_cvstep)         = train_time; % current training set
                    cfg.design.test_timepoint_per_step(1, full_cfg_cvstep)          = test_time; % current test set
                end
            end 
        end
    end
end

%% Sorting design before saving (optional)

% Does not lead to speedup, but visually look clearer 
% Sorting for speedup (without changing set entries) is done by TDT

% cfg.design = sort_design(cfg.design);

%% Return design

design = cfg.design; % return design only (only the design field should have been changed here)