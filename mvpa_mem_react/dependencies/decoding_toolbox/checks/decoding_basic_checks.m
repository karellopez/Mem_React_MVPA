% function [cfg, n_files, n_steps] = decoding_basic_checks(cfg,output_arguments)
%
% This is a subfunction of decoding and an integral part of the decoding
% toolbox. This function should not be called directly. It fulfills two 
% purposes: First, it carries out a lot of checks that make sure the user  
% specified the right options and that the decoding analysis can be 
% carried out without later errors happening in the middle of everything. 
% Second, some parameters are set that are later needed.

% History 
% Kai (2022/08/15): Added check if min0max1global scaling can be used for
%   libsvm, added option cfg.scale.force_libsvm_no_scaling
% Martin (2016/06/05): Added compatibility with AFNI
% Kai (2015/21/08): Automatic check for mask.img if not provided
% Martin (2014/16/06): Allowing only Matlab versions > 7.3 (all others have
%   not been tested)
% Martin (2014/01/24): Changed check for writing results, results are now 
%   overwritten at end of processing, not beginning
% Martin (2014/01/07): Externalized function from decoding.m

function [cfg, n_files, n_steps] = decoding_basic_checks(cfg,output_arguments)

% Display image access software that is used
dispv(1, 'Image access with: %s',cfg.software);

if strcmpi(cfg.software,'AFNI')
    warningv('DECODING_BASIC_CHECKS:AFNI_BETA_MODE','AFNI support in TDT is still in beta mode. Use with care and report bugs!')
end

% Check Matlab version
vers = sscanf(version,'%d.%d');
if vers(1) < 7 || (vers(1)==7 && vers(2) < 3)
    error(['Your Matlab version is older than 7.3. We did not test earlier \n',...
           'versions, so we strongly recommend not to use them. If you still \n',...
           'want to proceed, deactivate this error message manually in \n',...
           'decoding_basic_checks.m, but also remove the -v7.3 flag in decoding_write_results.m'])
end

% Warn if default directory is used
if cfg.results.write > 0
    default_cfg = decoding_defaults;
    if strcmp(cfg.results.dir, default_cfg.results.dir)
        warningv('decoding_basic_checks:default_result_dir', 'Saving results to default location cfg.results.dir: %s. We strongly recommend to set cfg.results.dir to a custom diretory if you want to store the data.\nHere the command to do so: cfg.results.dir = ''YOURDIR''.', cfg.results.dir)
    end
end

% Display what data is saved
switch cfg.results.write
    case 0
        dispv(1,'Results are not saved to harddisk.')
    case 1
        dispv(1,'Results are saved as .mat-file to %s',cfg.results.dir)
    otherwise
        dispv(1,'Results are saved as .mat-files and as brain volumes to %s',cfg.results.dir)
end

% File check here not necessary anymore.
% Moved the check to when a file function is used for the first time.
% - Kai
% check_software(cfg);

% check if design exists, and create if it doesn't
field_names = {'label','train','test','set'};
missing = [1 1 1 1];
for i_field = 1:length(field_names)
    if isfield(cfg.design,field_names{i_field})
        missing(i_field) = 0;
    end
end

if any(missing) % if only some or no fields for a design exist
    if isfield(cfg.design,'function') && isfield(cfg.design.function,'name') % create design with passed method
        if ~all(missing) % throw warning if some fields exist, but others not
            warningv('DECODING_BASIC_CHECKS:MissingFieldsInDesignReplaced','Some fields for design matrix were missing. Design is now created from scratch, using the method %s.',cfg.design.function.name)
        end
        fhandle = str2func(cfg.design.function.name); % create design
        cfg.design = feval(fhandle,cfg);
    else % throw error if no method has been passed and design incomplete
        error('Design is missing or incomplete. Either create design in advance or pass method to create design (see ''help decoding'')');
    end
end

% Set function handle for classifier here (saves time to do only once)
if ~isfield(cfg.decoding,'fhandle_train') && ~isfield(cfg.decoding,'fhandle_test')
        cfg.decoding.fhandle_train = str2func([cfg.decoding.software '_train']); % this format allows variable input
        cfg.decoding.fhandle_test = str2func([cfg.decoding.software '_test']); % this format allows variable input
else
    % Run quick test that method is the same for both:
    if ~strcmpi(func2str(cfg.decoding.fhandle_train),[cfg.decoding.software '_train']) || ...
       ~strcmpi(func2str(cfg.decoding.fhandle_test),[cfg.decoding.software '_test'])
       warningv('DECODING_BASIC_CHECKS:decoding_fhandle_name_mismatch',...
           'Mismatch between cfg.decoding.software and cfg.decoding.fhandle_train / cfg.decoding.fhandle_test. Getting info from cfg.decoding.software and discarding settings!')
       cfg.decoding.fhandle_train = str2func([cfg.decoding.software '_train']); % this format allows variable input
       cfg.decoding.fhandle_test = str2func([cfg.decoding.software '_test']); % this format allows variable input
    end
end

% Set function handle for classifier in parameter_selection
if ~strcmpi(cfg.parameter_selection.method,'none') && (~isfield(cfg.parameter_selection.decoding,'fhandle_train') || ~isfield(cfg.parameter_selection.decoding,'fhandle_test'))
    cfg.parameter_selection.decoding.fhandle_train = str2func([cfg.parameter_selection.decoding.software '_train']); % this format allows variable input
    cfg.parameter_selection.decoding.fhandle_test = str2func([cfg.parameter_selection.decoding.software '_test']); % this format allows variable input
end

% Set function handle for classifier in feature_selection
if ~strcmpi(cfg.feature_selection.method,'none') && (~isfield(cfg.feature_selection.decoding,'fhandle_train') || ~isfield(cfg.feature_selection.decoding,'fhandle_test'))
    cfg.feature_selection.decoding.fhandle_train = str2func([cfg.feature_selection.decoding.software '_train']); % this format allows variable input
    cfg.feature_selection.decoding.fhandle_test = str2func([cfg.feature_selection.decoding.software '_test']); % this format allows variable input
end

% Make sure that all labels are integers when the method is classification or classification_kernel
method = cfg.decoding.method;
if (strcmpi(method,'classification') || strcmpi(method,'classification_kernel')) && any(cfg.design.label(:)~=round(cfg.design.label(:)))
    error(['cfg.decoding.method = %s, but not all labels are integers (i.e. whole numbers). ',...
           'Change labels to whole numbers if you want to run %s. If you want to run a regression, set the method to ''regression''. ',...
           'In that case don''t forget to change cfg.results.output = ''corr'' or ''zcorr''.'],method,method)
end

% try the simplest decoding possible (only if libsvm is used)
if strcmpi(cfg.decoding.software,'libsvm')
    [working, libsvm_path] = check_libsvm(cfg);
    if ~working
        error('libsvm does not seem to work with the current parameters (Path: %s)', libsvm_path)
    else
        dispv(2, 'Checked that libsvm works with the current parameters')
        dispv(2, 'Using libsvm in: %s', libsvm_path)
    end
end

% Make outputs cell array
if ischar(cfg.results.output)
    cfg.results.output = num2cell(cfg.results.output,2);
end

% For feature transformation, only one of these fields can be entered, not both
if isfield(cfg.feature_transformation,'n_vox') && isfield(cfg.feature_transformation,'critical_value')
    error('It is only possible to provide either the field cfg.feature_transformation.n_vox or the field cfg.feature_transformation.critical_value, but not both')
end

% Check if kernel method is used
use_kernel = ~isempty(strfind(cfg.decoding.method, '_kernel'));
cfg.decoding.use_kernel = use_kernel;
if use_kernel
    dispv(1, 'Using a "_kernel" decoding method.')
    dispv(2, sprintf('This means that the kernel is only calculated once for each voxel/ROI,\nand then a submatrix of the kernel is passed to training and test methods \ninstead of the data. This might increase speed, but does not allow all\nmethods of TDT to be selected'))
else
    dispv(2, 'Using normal method')    
end

% Using a precomputed kernel doesn't work for scaling across or separate
if use_kernel && (strcmpi(cfg.scale.estimation,'across') || strcmpi(cfg.scale.estimation,'separate'))
    newmethod = strrep(cfg.decoding.method,'_kernel','');
    str = sprintf(['Use of scaling method ''%s'' and decoding method ''%s'' does not make sense, because a kernel must calculated in this case in every step anyway (i.e. there is no benefit in using a kernel over the classical methods without kernels).',...
                   'Method is now reverted to ''%s'' (which will be slower).'],cfg.scale.estimation,cfg.decoding.method,newmethod);
    warningv('DECODING_BASIC_CHECKS:KernelAndScaling',str)
    cfg.decoding.method = newmethod;
    cfg.decoding.use_kernel = 0;
end

% Check if libsvm is used and if it can be used with scaling, warn else
if strcmpi(cfg.decoding.software, 'libsvm') && strcmpi(cfg.scale.method, 'none')
    dispv(1, 'libsvm is used (cfg.decoding.software = ''libsvm'') without scaling cfg.scale.method = ''none''. This can be very slow. Checking if the transform result methods (set in cfg.result.output) are ok with min0max1global scaling..')
    assert(iscell(cfg.results.output), 'internal error: cfg.results.output should be a cell at this position')
    if isempty(cfg.results.output)
        warningv('decoding_basic_checks:autoscaling_not_ok_no_measures', ...
            'cfg.results.output is empty (i.e. no measures are passed). Because it might be that autoscaling with min0max1global is not ok. This will lead to an error. Set cfg.scale.method if this is ok (see help decoding_scale_data)')
        scaling_ok = false; % automatic scaling not ok if 
    elseif isfield(cfg.scale, 'force_libsvm_no_scaling') && cfg.scale.force_libsvm_no_scaling
        dispv(1, 'decoding_basic_checks: keeping cfg.scale.method = ''none'' because cfg.scale.force_libsvm_no_scaling = true');
        scaling_ok = false; % automatic scaling not ok if 
    else
        scaling_ok = true; % will be set to false as soon as one method is not on the positive list
        scaling_ok_measures = {
            'accuracy', 'accuracy_minus_chance', ...
            'decision_values', 'predicted_labels', 'true_labels', ...
            'sensitivity', 'sensitivity_minus_chance', ...
            'specificity', 'specificity_minus_chance', ...
            'balanced_accuracy', 'balanced_accuracy_minus_chance', ...
            'dprime', 'loglikelihood', ...
            'AUC', 'AUC_minus_chance', 'AUC_pairwise', 'AUC_pairwise_minus_chance', ...
            'AUC_matrix', 'AUC_matrix_minus_chance', ...
            'corr', 'zcorr', ... % end measures decoding_transform_results
            ... % start transres_ functions
            'accuracy_matrix', 'accuracy_matrix_minus_chance', ...
            'accuracy_pairwise', 'accuracy_pairwise_minus_chance', ...
            'binomial_probability', ... 
            'confusion_matrix', 'confusion_matrix_plus_undecided', ...
            'MSS_diff', ...
            'ninputdim', ...
            'predicted_and_true_labels_multitarget', 'predicted_labels_multitarget', ...
            'rsa_beta', 'RSA_dist_corr', 'RSA_dist_euclidean', ...
            'signed_decision_values'
            };
        scaling_not_ok_measures = {
            ... % no measures in decoding_transform_results have problems with global scaling
            'SVM_pattern', 'SVM_pattern_alldata', ... % transres functions
            'SVM_weights', 'SVM_weights_plusbias', ...
            'primal_SVM_weights', 'primal_SVM_weights_nobias', ...
            };
        
        % special case: 'model_parameters'
        % common measure, ok to use scaling if the parameters are applied 
        % directly, which is common, but not ok  if parameters should be 
        % stored (then the scaling parameters need to be stored and 
        % applied, too, which is not implemented yet)
        % Current solution: 
        % Changing to min0max1global scaling is accepted, if:
        %     cfg.scale.check_datatrans_ok = true
        % Scaling is forced to 'none', if
        %     cfg.scale.force_libsvm_no_scaling = true
        %   and 
        %     cfg.scale.IKnowThatLibsvmCanBeSlowWithoutScaling = true
        % If cfg.scale.IKnowThatLibsvmCanBeSlowWithoutScaling is not true,
        % and scaling is applied, an error occurs.
        
        for measure_ind = 1:length(cfg.results.output)
            curr_measure = cfg.results.output{measure_ind};
            % special case: mode paramters
            if strcmpi('model_parameters', curr_measure)
                if isfield(cfg.scale, 'force_libsvm_no_scaling') && cfg.scale.force_libsvm_no_scaling
                    dispv(1, 'decoding_basic_checks: ')
                dispv(1, 'decoding_basic_checks: ', ...
                    ['cfg.results.output{' num2str(measure_ind) '} = ''model_paramters'': ' ...
                    'Because you set cfg.scale.force_libsvm_no_scaling = true, no data scaling will be applied. ' ...
                    'Note: To acknowledge that this can be slow, set also ' ...
                    'cfg.scale.IKnowThatLibsvmCanBeSlowWithoutScaling = true. ' ...
                    'To allow scaling for models, set cfg.scale.check_datatrans_ok = true and cfg.scale.force_libsvm_no_scaling = false']);
                end
            end
            
            if any(strcmpi(scaling_ok_measures, curr_measure))
                % ok for scaling, nothing to do
            elseif any(strcmpi(scaling_not_ok_measures, curr_measure))
                % not ok for scaling - inform and fail
                warningv(['decoding_basic_checks:autoscaling_not_ok_measure_' curr_measure], ...
                    ['Transformation measure cfg.results.output{' num2str(measure_ind) '} = ' ...
                     '' curr_measure ''' is NOT suitable for min0max1scaling. Check error message below']);
                scaling_ok = false;
            else
                % unclear if scaling ok - try to ask function
                try
                    if ischar(curr_measure) % call method if is it s char
                        fhandle = str2func(['transres_' curr_measure]);
                        output = feval(fhandle, 'scaling_libsvm_min0max1global_ok');
                    elseif isobject(curr_measure) % call method if it is an object
                        output = curr_measure.apply('scaling_libsvm_min0max1global_ok');
                    end
                    if output == false
                        scaling = false;
                        if isobject(curr_measure)
                            curr_measure = 'PASSED_OBJECT';
                        end
                        warningv(['decoding_basic_checks:autoscaling_not_ok_measure_' curr_measure], ...
                            ['Transformation measure cfg.results.output{' num2str(measure_ind) '} = ' ...
                            '' curr_measure ''' is NOT suitable for min0max1scaling. Check error message below']);
                    end
                    continue % function returned state, jump to next measure
                end
                % could not determine if scaling is ok or not - inform and fail
                warningv(['decoding_basic_checks:autoscaling_unclear_measure_' curr_measure], ...
                    ['Unclear if transformation measure cfg.results.output{' num2str(measure_ind) '} = ' ...
                     '' curr_measure ''' is suitable for min0max1scaling. ' ...
                     'If you developed the measure and it is suitable for scaling, you can avoid this error by ensuring that calling transres_' curr_measure '(''scaling_libsvm_min0max1global_ok'') returns true or false if it is not ok (or if you use an object as transres, the same with a call to yourobject.method()).' ...
                     'For more info check error message below']);
                scaling_ok = false;
            end
        end
    end
    
    
    if scaling_ok
        cfg.scale.method = 'min0max1global';
            warningv('decoding_basic_checks:setting_default_scaling_for_libsvm', ...
               ['NOTE: This is a new warning introduced in TDT version 3.999F 2022/08/01. ' ...
                'PROBLEM: Libsvm can be VERY slow for some datasets (e.g. some subjects or folds) when data is not scaled (For more, see help decoding_scale_data).' ...
                'SOLUTION: To prevent this, employ a scaling method. libsvm recommends ''min0max1global'' for ''all'' data.'...
                'If no internal parameters of the model (e.g. "weights" of a classifier) should be retrieved, this yields the same results.'...
                'We did not detect any method that returns model weights, and therefore set cfg.scale.method = ''min0max1global''']);
        if strcmpi(cfg.scale.estimation, 'none')
            dispv(1, 'Changing cfg.scale.estimation from ''none'' to ''all'' to allow min0max1global scaling')
            cfg.scale.estimation = 'all'; % 
        elseif strcmpi(cfg.scale.estimation, 'all')
            % all fine
        else
            warningv('decoding_basic_checks:unexpected_estimation_val_when_scaling_for_libsvm', ...
            ['When setting cfg.scale.method to ''min0max1global'' to prevent slow libsvm performance, cfg.scale.estimation has been set and is not ' ...
                '''none'' or ''all'' as expected, but ''' cfg.scale.estimation '''. We assume you know what you do ']);
        end
    else
        if ~isfield(cfg.scale, 'IKnowThatLibsvmCanBeSlowWithoutScaling')
            error('decoding_basic_checks:setting_default_scaling_for_libsvm', ...
               ['NOTE: This is a warning was introduced in TDT version 3.999E 2021/09/09 and adapted to min0max1global in 3.999F 2022/08/01. ' ...
                'PROBLEM: Libsvm can be VERY slow for some datasets (e.g. some subjects or folds) when data is not scaled. ' ...
                'SOLUTION: To prevent this, employ a scaling method. libsvm recommends ''min0max1global'' for ''all'' data.'...
                'To do so, add the following before you call decoding(cfg): cfg.scale.method = ''min0max1global''; cfg.scale.estimation = ''all''; (For more, see help decoding_scale_data). ' ...
                'If you really dont want to do any scaling, set cfg.scale.IKnowThatLibsvmCanBeSlowWithoutScaling = 1 (but be prepared - it can be SLOW). ' ...
                'NOTE: If you have many old scripts, consider changing this error to a warning in decoding_basic_checks.m, and everything will work as it used to.'])
        end
    end
end

if use_kernel && strcmpi(cfg.feature_transformation.estimation,'across')
    newmethod = strrep(cfg.decoding.method,'_kernel','');
    str = sprintf(['Use of cfg.feature_transformation.estimation = ''across'' and decoding method ''%s'' is not possible at the moment. ',...
                   'Method is now reverted to ''%s'' (which will be slower).'],cfg.decoding.method,newmethod);
    warningv('DECODING_BASIC_CHECKS:KernelAndFeatureTransformation',str)
    cfg.decoding.method = newmethod;
    cfg.decoding.use_kernel = 0;
end

% Using feature selection with a kernel method in the main function doesn't make sense
if use_kernel && ~strcmpi(cfg.feature_selection.method,'none')
    newmethod = strrep(cfg.decoding.method,'_kernel','');
    str = sprintf(['Use of feature selection together with decoding method ''%s'' in the main function makes processing slower. ',...
                   'Method is now reverted to ''%s''.'],cfg.decoding.method,newmethod);
    warningv('DECODING_BASIC_CHECKS:KernelAndFeatureSelection',str)
    cfg.decoding.method = newmethod;
    cfg.decoding.use_kernel = 0;
end

% Using the kernel can disagree with the parameters set manually for
% decoding, checking this for libsvm and the default linear kernel
if use_kernel && strcmpi(cfg.decoding.software,'libsvm')
    fstr = func2str(cfg.decoding.kernel.function);
    fstr2 = strfind(cfg.decoding.train.classification.model_parameters,'-t 0'); % check for linear kernel
    if strcmpi(fstr,'@(X,Y)X*Y''') && isempty(fstr2)
        str = ['Using classification with a linear kernel, but manual settings of classification are set to nonlinear. ',...
            'We want to prevent you from making a mistake. Please set either cfg.decoding.method = ''classification'' or ',...
            'in cfg.decoding.train.classification.model_parameters, set -t 0'];
        error(str)
    end
end

if ~strcmpi(cfg.feature_selection.method,'none')
    warningv('DECODING_BASIC_CHECKS:FeatureSelectionIsBeta','We have only little feedback from users about feature selection so far. Running in beta stage!')
    if ischar(cfg.feature_selection.results.output)
        cfg.feature_selection.results.output = num2cell(cfg.feature_selection.results.output,2);
    end
    if isfield(cfg.feature_selection.results,'output') && numel(cfg.feature_selection.results.output)>1
        error(['More than one output selected in nested CV for feature selection. ',...
            'Change field ''cfg.feature_selection.results.output'' to one entry only.'])
    end
end

if ~strcmpi(cfg.parameter_selection.method,'none')
    if ischar(cfg.parameter_selection.results.output)
        cfg.parameter_selection.results.output = num2cell(cfg.parameter_selection.results.output,2);
    end
    if isfield(cfg.parameter_selection.results,'output') && numel(cfg.parameter_selection.results.output)>1
        error(['More than one output selected in nested CV for parameter selection. ',...
            'Change field ''cfg.parameter_selection.results.output'' to one entry. only.'])
    end
end

[n_files, n_steps] = size(cfg.design.train);

dispv(1,'Performing %i decoding steps for %i files', n_steps, n_files)

% check that number of files = number of rows in cfg.design
if n_files ~= size(cfg.design.train, 1)
    error('Number of files in cfg.files (%i) does not correspond to number of rows in cfg.design.train', n_files, size(cfg.design.train, 1))
end

if n_files ~= size(cfg.design.test, 1)
    error('Number of files in cfg.files (%i) does not correspond to number of rows in cfg.design.test', n_files, size(cfg.design.train, 1))
end

if ~isequal(size(cfg.design.train), size(cfg.design.test))
    error('Size mismatch: ~isequal(size(cfg.design.train), size(cfg.design.test))')
end

if isfield(cfg, 'multitarget') && cfg.multitarget
    warningv('decoding_basic_checks:multitarget_no_check', 'Multitarget mode detected. Checks for labels currently not implemented, skipping this.')
    warningv('decoding_basic_checks:multitarget_n_cond_nan', 'Multitarget mode detected. Setting cfg.design.n_cond = nan;.')
    cfg.design.n_cond = nan;
    warningv('decoding_basic_checks:multitarget_n_cond_per_step', 'Multitarget mode detected. Setting cfg.design.n_cond_per_step = nan;.')
    cfg.design.n_cond_per_step = nan;
else
    % get number of conditions present in decoding
    cfg.design.n_cond = length(unique(cfg.design.label(cfg.design.train | cfg.design.test))); % all used labels

    % get number of *used* conditions (i.e. labels) for each run separately
    n_unique_labels = zeros(1,n_steps);
    unique_labels = cell(1,n_steps);
    for i_step = 1:n_steps
        curr_label = cfg.design.label(:,i_step);
        unique_labels{i_step} = unique(curr_label(cfg.design.train(:,i_step) | cfg.design.test(:,i_step)));
        n_unique_labels(i_step) = length(unique_labels{i_step});
    end
    % at the same time make sure that the number is always the same (it is possible that
    % different labels are used as long as the number of labels remains the
    % same)
    if ~strcmpi(cfg.decoding.method,'regression')
        if ~all(n_unique_labels == n_unique_labels(1))
            error('Number of used labels varies across decoding steps which prevents comparing results across steps. If multiple sets are used, run them separately.')
        else
            diff_unique_labels = diff([unique_labels{:}],1,2);
            if any(diff_unique_labels(:)) % if any run contains different labels
                warningv('DECODING_BASIC_CHECKS:more_than_two_labels',...
                    'More than two labels are used, but not all labels are used in each run (e.g. in run 1 labels A and B and in run 2 labels A and C). Make sure this has been intended!')
            end
        end
    end
    cfg.design.n_cond_per_step = n_unique_labels(1);


    problem = 0;
    for i_step = 1:n_steps
        curr_train = cfg.design.train(:,i_step);
        curr_test = cfg.design.test(:,i_step);
        curr_label = cfg.design.label(:,i_step);
        if length(unique(curr_label(logical(curr_train)))) == 1
            error('Training data in decoding step %i contains only one label, but needs at least two.',i_step)
        end
        if length(unique(curr_label(logical(curr_test)))) == 1
            problem = problem+1;
        end
    end
    if problem && n_steps == 1
        warningv('DECODING_BASIC_CHECKS:TestDataOnlyOneLabel',...
            ['Test data in %i steps contains only one label and there is only ',...
             'one decoding step. This might be a problem when using correlation, ',...
             'AUC, sensitivity, specificity and similar measures!'],problem)
    end
end

% Get number of sets
n_sets = length(unique(cfg.design.set));

% Run check that results are returned setwise when running a permutation design
try
    if isfield(cfg.design.function.permutation) && cfg.results.setwise == 0
        error('Using a permutation design with cfg.results.setwise == 0. This will make the results uninterpretable. Please set cfg.results.setwise = 1')
    end
end

if strcmpi(cfg.scale.method,'none') && ~strcmpi(cfg.scale.estimation,'none')
    error(['Scaling method is ''none'', but estimation type is ''' cfg.scale.estimation '''. Unknown if you want to scale or not! Set both to ''none'' or both to a different value than ''none''!'])
%     warningv('DECODING_BASIC_CHECKS:DisagreeingScalingMethodAndEstimation',['Scaling method is ''none'', but estimation type is ''' cfg.scale.estimation ''', changing type to ''none'''])
end

% Check if user flipped input to decoding_scale_data
if any(strcmpi({'z','min0max1','cov'},cfg.scale.estimation))
   error('cfg.scale.estimation = ''%s'', but this value is a valid input only to cfg.scale.method. Please check if you flipped the input.',cfg.scale.estimation) 
end
if any(strcmpi({'all','across','separate'},cfg.scale.method))
   error('cfg.scale.method = ''%s'', but this value is a valid input only to cfg.scale.estimation. Please check if you flipped the input.',cfg.scale.method) 
end

% check if masks exist, and maybe correct it. Otherwise set it to "auto"
if isfield(cfg.files, 'mask') && ~isempty(cfg.files.mask)
    if ischar(cfg.files.mask)
        cfg.files.mask = num2cell(cfg.files.mask,2);
    end
else % mask not specified, check if all input files come from one directory and if this contains a mask.img/.nii/+orig.BRIK/+acpc.BRIK/+tlrc.BRIK, else use all voxels
    set_auto = 1; % by default, set mask to 'all voxels'
    try % if something doesn't work, we don't care and just use the auto version below
        dispv(1, 'No mask specified in cfg.files.mask, checking if all images are from the same directory')
        [p1, fn, ext] = fileparts(cfg.files.name{1}); %#ok<*NASGU,*ASGLU> % directory of first file
        % check if all other files are in the same directory
        all_from_same_directory = 1; % init
        for file_ind = 2:length(cfg.files.name)
            [p, fn, ext] = fileparts(cfg.files.name{file_ind});
            if ~strcmp(p1, p) % if directory is not the same as for first file, break
                dispv(2, 'Not all inputfiles are from the same directory, so not checking for mask.img/.nii/+orig.BRIK/+acpc.BRIK/+tlrc.BRIK')
                all_from_same_directory = 0;
                break
            end
        end
        
        if all_from_same_directory
            % all images are from the same directory
            % check if the directory contains a mask.img/.nii/.BRIK
            potential_mask_img = fullfile(p1, 'mask.img');
            if ~exist(potential_mask_img, 'file') % it might also be a .nii file
                potential_mask_img = fullfile(p1, 'mask.nii');
            end
            if ~exist(potential_mask_img, 'file') % it might also be a +orig.BRIK file
                potential_mask_img = fullfile(p1, 'mask+orig.BRIK');
            end
            if ~exist(potential_mask_img, 'file') % it might also be a +acpc.BRIK file
                potential_mask_img = fullfile(p1, 'mask+acpc.BRIK');
            end
            if ~exist(potential_mask_img, 'file') % it might also be a +tlrc.BRIK file
                potential_mask_img = fullfile(p1, 'mask+tlrc.BRIK');
            end
            if exist(potential_mask_img,'file')
                cfg.files.mask = potential_mask_img;
                dispv(1, 'All files in same directory that contains a mask.img/.nii/+orig.BRIK/+acpc.BRIK/+tlrc.BRIK. Setting cfg.files.mask=%s', cfg.files.mask)
                set_auto = 0; % do nothing more 
            else
                dispv(2, 'All files from same directory but no mask.img/.nii/+orig.BRIK/+acpc.BRIK/+tlrc.BRIK in this directory, so switching to auto. Directory was: %s', p1)
            end
        end
    catch %#ok<*CTCH>
        warningv('mask_img_detection_failed', 'Something did not work with automatic detection of mask.img/.nii/+orig.BRIK/+acpc.BRIK/+tlrc.BRIK, using all voxels')
    end

    if set_auto % set it to auto
        dispv(1, 'No mask file detected, using all voxels')
        cfg.files.mask = {'all voxels'}; % will generate a mask later (using all voxels)
    end
end

results_out_flag = output_arguments >= 1; % flag showing whether the results are returned from the function

if cfg.results.write == 0 && ~results_out_flag
    error('''Write results'' set to 0, but results are not returned either. Change ''write results'' to >0 or return results as output')
end

if strcmpi(cfg.parameter_selection.method,'none') && isfield(cfg.parameter_selection,'parameters')
    error('Field ''cfg.parameter_selection.parameters'' exists, but ''cfg.parameter_selection.method = ''none''!')
end

% parameter_selection: check if for libsvm or liblinear, parameters are passed in format '-<string>'
if isfield(cfg.parameter_selection,'parameters') && (strcmpi(cfg.decoding.software,'libsvm') || strcmpi(cfg.decoding.software,'liblinear'))
    tmp = cfg.parameter_selection.parameters;
    if ~iscell(tmp), tmp = num2cell(tmp,2); end
    ok = true;
    for i_tmp = 1:length(tmp)    
        if ~ischar(tmp{i_tmp})
            ok = false;
        else
            ok = ok & strcmp(tmp{i_tmp}(1),'-');
        end
        if ~ok
        error('Error passing parameters in cfg.parameter_selection.parameters. For libsvm or liblinear, parameter needs to be passed in the format -<string>, e.g. ''-c'' or ''-g'' ')
        end
    end
end


% Checking for independence of training and test data
if isfield(cfg.design,'train_eq_test') && cfg.design.train_eq_test == 1 % this may avoid the call to the non-independence check, but should never be set manually!
    warningv('DECODING_BASIC_CHECKS:TRAINEQTEST','cfg.design.train_eq_test == 1. You are assuming that training data equals test data. If this assumption is wrong, then your results are not interpretable!')
else
    if any(cfg.design.train(:) ~= 0 & cfg.design.test(:) ~=0)
        disp('Positions of Entries in Training- & Testset:')
        disp(cfg.design.train ~= 0 & cfg.design.test ~= 0)
        if isfield(cfg.design,'nonindependence')
            check1 = strcmpi(cfg.design.nonindependence,'ok');
        else
            check1 = 0;
        end
        check2 = strfind(cfg.results.output,'SVM_weights');
        check2 = [check2 strfind(cfg.results.output,'SVM_pattern')];
        try check2 = cell2mat(check2); end %#ok<*TRYNC>
        check2 = ~isempty(check2);
        if check1 || check2
            warningv('DECODING_BASIC_CHECKS:Nonindependence','Training and test data are not independent. If you return classification results, they cannot be interpreted!');
        else
            error(['Trainingset & Testset are not independent! Some entries from the training set are also used in the testset! Please check!',...
                ' If you really know what you are doing, set cfg.design.nonindependence = ''ok'' in your script.'])
        end
    else
        dispv(2,'  Check for double entries in Training- & Testset: No double entries found.')
    end
end

% Check if training data is balanced (problematic!) and test data (may matter)
if isfield(cfg, 'multitarget') && cfg.multitarget
    warningv('decoding_basic_checks:multitarget_no_check_imbalance', 'Multitarget mode detected. Checks for labels currently not implemented. Skipping check_imbalance().')
else
    check_imbalance(cfg);
end
    
if ischar(cfg.files.name)
    cfg.files.name = num2cell(cfg.files.name,2);
    warningv('DECODING_BASIC_CHECKS:FileNamesStringNotCell','File names provided as string, not as cell matrix. Converting to cell...')
end

if length(cfg.files.name) ~= length(unique(cfg.files.name))
    if isfield(cfg, 'basic_checks') && isfield(cfg.basic_checks, 'DoubleFilenameEntriesOk') && cfg.basic_checks.DoubleFilenameEntriesOk == 1
        % explicitly acknowledge, warning only
        warningv('DECODING_BASIC_CHECKS:DoubleFilenameEntries','Double filename entries in cfg.files.name. No guarantee, that training and test sets are independent!!!')
    else
        error('DECODING_BASIC_CHECKS:DoubleFilenameEntries','Double filename entries in cfg.files.name. No guarantee, that training and test sets are independent!!! Set cfg.basic_checks.DoubleFilenameEntriesOk = 1 to allow double file names.')
    end
else
    dispv(2,'  Check for double names in cfg.files.name: No double entries found.')
end

if ~strcmpi(cfg.scale.method,'none') && numel(cfg.scale.cutoff) ~= 2
    error('Wrong number of entries for field ''cfg.scale.cutoff''.')
end

if strcmpi(cfg.feature_selection.method,'filter') && strcmpi(cfg.feature_selection.filter,'external') 
    if ~isfield(cfg.feature_selection,'external_fname')
        error('Feature selection method ''external'' was chosen, but no file name was provided in cfg.feature_selection.external_fname.')
    else
        if ischar(cfg.feature_selection.external_fname)
            cfg.feature_selection.external_fname = num2cell(cfg.feature_selection.external_fname,2);
            warningv('DECODING_BASIC_CHECKS:convert_to_cell','Converting cfg.feature_selection.external_fname from character to cell. Try providing cell arrays in the future')
        end
        n_external = length(cfg.feature_selection.external_fname);
        if  n_external ~= 1 && n_external ~= n_steps
            error('Number of external images need to be 1 or one for each decoding step (i.e. %i), but is %i',n_steps,n_external)
        end
    end
end

if cfg.results.write

    dir_output = cfg.results.dir; % results directory
    if ~exist(dir_output, 'dir'), mkdir(dir_output); end
    dispv(2,'Creating output path at %s',dir_output)
    
    n_outputs = length(cfg.results.output);
    if ~isfield(cfg.results,'resultsname')
        for i_output = 1:n_outputs
            outputname = cfg.results.output{i_output};
            % create file names for results that are written
            cfg.results.resultsname(i_output) = { sprintf('%s_%s',cfg.results.filestart,outputname) };
        end
    end

    for i_output = 1:n_outputs

        % Check if it is ok and possible to overwrite existing files

        ext = {'.img','.hdr','.mat','.nii','+orig.BRIK','+orig.HEAD','+acpc.BRIK','+acpc.HEAD','+tlrc.BRIK','+tlrc.HEAD'};
        if cfg.results.write == 2, ext = {'.mat'}; end % check matfile only
        for ext_ind = 1:length(ext)
            if isfield(cfg.design,'function') && isfield(cfg.design.function,'permutation')
                % do not run check when we are running a permutation test
            else
                % create full path for results that are written
                output_fname = [fullfile(dir_output,cfg.results.resultsname{i_output}) ext{ext_ind}];
                % check if it is possible to write
                check_write(output_fname,cfg.results.overwrite)
            end
            
            % If setwise check all files
            if cfg.results.setwise && n_sets > 1
                set_numbers = unique(cfg.design.set);
                for i_set = 1:n_sets
                    output_fname = fullfile(dir_output,sprintf('%s_set%i%s', cfg.results.resultsname{i_output}, set_numbers(i_set), ext{ext_ind}));
                    check_write(output_fname,cfg.results.overwrite)
                end
            end
            
        end
    end
    
end


%% Subfunction: Check for unbalanced training data (not allowed) and unbalanced test data (potentially problematic)
function check_imbalance(cfg)
dispv(2, 'Checking for imbalances in cfg.design.train')
for decoding_step = 1:size(cfg.design.train, 2)
    curr_labels = cfg.design.label(:, decoding_step);
    curr_train_labels = curr_labels(cfg.design.train(:,decoding_step) == 1);
    curr_test_labels = curr_labels(cfg.design.test(:,decoding_step) == 1);
    unique_train_labels = unique(curr_train_labels);
    unique_test_labels = unique(curr_test_labels);
    n_each_train_label = zeros(length(unique_train_labels),1);
    n_each_test_label = zeros(length(unique_test_labels),1);
    for label_ind = 1:length(unique_train_labels)
        n_each_train_label(label_ind) = sum(curr_train_labels == unique_train_labels(label_ind));
    end
    for label_ind = 1:length(unique_test_labels)
        n_each_test_label(label_ind) = sum(curr_test_labels == unique_test_labels(label_ind));
    end
    % Run check for training data
    if any(diff(n_each_train_label) ~= 0)
        message_str = sprintf('Unbalanced training data detected in cfg.design.train(:, %i).', decoding_step);
        if isfield(cfg.design, 'unbalanced_data') && strcmpi(cfg.design.unbalanced_data, 'ok')
            warningv('DECODING:CheckUnbalancedDataOk', [message_str, ' You decided this is ok, because cfg.design.unbalanced_data = ''ok''']);
        else
            error('DECODING:CheckUnbalancedDataOk', [message_str, ' If this is ok, set cfg.design.unbalanced_data = ''ok'''])
        end
    end
    % Run check for test data
    if any(diff(n_each_test_label) ~= 0)
        message_str = sprintf('Unbalanced test data in cfg.design.test(:, %i).', decoding_step);
        dispv(1,message_str)
        test = [strfind(cfg.results.output,'accuracy') strfind(cfg.results.output,'accuracy_minus_chance')];
        if any([test{:}])
            warningv('DECODING:CheckUnbalancedTestData','You are returning accuracy or accuracy_minus_chance. These results may be uninterpretable due to test data imbalance.')
        else
            dispv(1,'This does not matter as long as your results measure returns unbiased results (e.g. balanced_accuracy).')
        end
    end
end


function check_write(output_fname,overwrite_flag)

if exist(output_fname,'file')
    if ~overwrite_flag
        error(['Resultfile %s already exists. Change filename or ',...
            'set cfg.results.overwrite = 1'],output_fname)
    else
        warningv('DECODING_BASIC_CHECKS:OverwritingExistingResultsfile',sprintf('Resultfile %s already existed. Overwriting at end of process...',output_fname))
    end
    
    % Get permissions and check if we can write
    [ignore,permissions] = fileattrib(output_fname); 
    if permissions.UserWrite ~=1
        error('Results cannot be written to %s \nCheck that you have writing permission.',output_fname)
    end
    
else
    % Check if it is possible to write
    temp = fopen(output_fname, 'w');
    if temp == -1, error('Results cannot be written to %s \nCheck that you have writing permission.',output_fname), end
    fclose(temp);
    delete(output_fname)
end