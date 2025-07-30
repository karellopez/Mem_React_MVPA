function results = mvpa_MR(study_mat_path, mask_file, out_dir, train_labels, train_runs, xclass_specs, varargin)
% MVPA_MR (Memory Reactivation)
% -------------------------------------------------------------------------
% Wrapper around the Decoding Toolbox that implements the complete analysis
% pipeline for one subject. The function performs two main steps:
%   1) Run leave-one-run-out cross‑validation within the specified training
%      set (for example DE_faces vs DE_scenes).
%   2) Train on the full training set and cross-classify on any number of
%      additional label/run combinations (e.g. AB, AC), evaluated run by run.
%
% All inputs are plain Matlab types so that the function can be called from
% a script or parfor loop.
%
% INPUTS (required)
%   study_mat_path : char/string. Path to *_study_data.mat (contains study_data struct)
%   mask_file      : char/string. Path to ROI/mask NIfTI
%   out_dir        : char/string. Output directory
%   train_labels   : cellstr/string array. Labels to use for CV training (e.g., {'DE_faces','DE_scenes'})
%   train_runs     : numeric vector. Runs to include in CV (e.g., 1:4)
%   xclass_specs   : cell array, each row describes one cross-class test:
%                    { test_labels(cell/str array), test_runs(vector), tag(char) }
%                    Example:
%                       xclass_specs = {
%                           {'AB_faces','AB_scenes'}, 1:4, 'AB';
%                           {'AC_faces','AC_scenes'}, 2:5, 'AC'
%                       };
%
% NAME-VALUE PAIRS (optional)
%   'BalanceTrain'  (false)  : balance faces/scenes in training set
%   'CfgTemplate'   ([])     : custom cfg (if empty, decoding_defaults)
%   'SavePNGs'      (true)   : save confusion matrices as PNGs
%   'Overwrite'     (true)   : overwrite existing results
%   'FixOldPath'    ({old,new}) : replace old root path by new inside study_data
%
% OUTPUT
%   results : struct with fields
%       .cv        -> CV results/cfg/confusion matrix
%       .xclass    -> struct per tag; each has cm_list/vec_list/mean_cm/mean_vec
%       .table     -> summary table
%
% -------------------------------------------------------------------------
% Author: Karelo López (2025)
% -------------------------------------------------------------------------

%% -------- Parse inputs --------
% --- Parse required and optional inputs ------------------------------------
p = inputParser;
addRequired(p,'study_mat_path',@(x) ischar(x)||isstring(x));
addRequired(p,'mask_file',     @(x) ischar(x)||isstring(x));
addRequired(p,'out_dir',       @(x) ischar(x)||isstring(x));
addRequired(p,'train_labels',  @(x) iscellstr(x)||isstring(x));
addRequired(p,'train_runs',    @(x) isnumeric(x) && isvector(x));
addRequired(p,'xclass_specs',  @(x) iscell(x));

addParameter(p,'BalanceTrain',false,@islogical);
addParameter(p,'CfgTemplate',[],@(x) isstruct(x)||isempty(x));
addParameter(p,'SavePNGs',true,@islogical);
addParameter(p,'Overwrite',true,@islogical);
addParameter(p,'TrainFilter',struct(),@isstruct);
addParameter(p,'FixOldPath',{},@(x) (iscell(x)&&numel(x)==2)||isempty(x));

parse(p,study_mat_path,mask_file,out_dir,train_labels,train_runs,xclass_specs,varargin{:});
opt = p.Results;

if ~exist(out_dir,'dir'); mkdir(out_dir); end

%% -------- Load study_data --------
% Each *_study_data.mat file contains a struct array 'study_data' with
% trial‑wise information and paths to beta images. Here we load it and
% optionally fix outdated path prefixes.
S = load(study_mat_path);
if ~isfield(S,'study_data')
    error('Variable "study_data" not found in %s', study_mat_path);
end
study_data = S.study_data;

% Optional path fix
if ~isempty(opt.FixOldPath)
    study_data = fix_paths_in_struct(study_data, opt.FixOldPath{1}, opt.FixOldPath{2});
end

% To table for convenient handling
T = struct2table(study_data);

% Beta files (flatten if nested)
betaFiles = T.beta_files;
if iscell(betaFiles) && numel(betaFiles)==1 && iscell(betaFiles{1})
    betaFiles = betaFiles{1};
end
betaFiles  = betaFiles(:);
conditions = string(T.category(:));
runs       = double(T.run(:));

% Filter valid paths
ok = ~cellfun(@isempty,betaFiles) & cellfun(@(p) exist(p,'file')==2, betaFiles);
betaFiles  = betaFiles(ok);
conditions = conditions(ok);
runs       = runs(ok);
Tbl        = T(ok,:);  % table aligned with filtered vectors

% Informative count of trials per condition at the raw level
if isempty(betaFiles)
    error('mvpa_MR:noTrials','No valid trials found after filtering beta files');
end
print_cond_counts(conditions,true(size(conditions)), 'Initial trials available');
print_run_cond_counts(conditions, runs, true(size(conditions)), 'Initial trials per run');

% Count trials for the requested training labels/runs before any user filters
train_mask_base = ismember(conditions, string(train_labels)) & ismember(runs, train_runs);
print_cond_counts(conditions, train_mask_base, 'Training set before TrainFilter');
print_run_cond_counts(conditions, runs, train_mask_base, 'Training per run before TrainFilter');

% Report how many trials exist for each testing specification before any
% additional filters. This helps verify that beta file loading worked
% for the cross-classification data as well.
for ii = 1:size(xclass_specs,1)
    t_lbls = string(xclass_specs{ii,1});
    t_runs = xclass_specs{ii,2};
    tg     = xclass_specs{ii,3};
    base_mask = ismember(conditions, t_lbls) & ismember(runs, t_runs);
    print_cond_counts(conditions, base_mask, sprintf('Test set %s before filters', tg));
    print_run_cond_counts(conditions, runs, base_mask, sprintf('Test per run before filters %s', tg));
end

% Convert trial condition strings ("*_faces" vs "*_scenes") into binary
% labels expected by the Decoding Toolbox. Faces -> +1, Scenes -> -1.
isFace = contains(conditions,'_faces');
labels = double(isFace);
labels(labels==0) = -1;

%% -------- Build cfg --------
% Start from the TDT defaults (or a user template) and set the basic
% parameters for an ROI‑based decoding analysis.
if isempty(opt.CfgTemplate)
    cfg = decoding_defaults;
else
    cfg = opt.CfgTemplate;
end
cfg.analysis         = 'ROI';
cfg.files.mask       = mask_file;
cfg.results.dir      = out_dir;
    cfg.results.output   = {'accuracy','accuracy_minus_chance', ...
                          'balanced_accuracy','AUC_minus_chance', ...
                          'confusion_matrix_plus_undecided', ...
                          'predicted_labels','true_labels', ...
                          'decision_values'};
cfg.scale.method     = 'min0max1';
cfg.scale.estimation = 'all';
cfg.plot_selected_voxels = 500;
if opt.Overwrite, cfg.results.overwrite = 1; end

%% -------- TRAIN: Cross-validation within train set --------
% Select the trials used for CV based on the requested labels and runs.
% Additional user-defined filters can further restrict the training set.
train_mask = ismember(conditions, string(train_labels)) & ismember(runs, train_runs);
% extra user-defined filters on training set
if ~isempty(fieldnames(opt.TrainFilter))
    keep_train = apply_filters(Tbl, opt.TrainFilter);
    train_mask = train_mask & keep_train;
end

% Optional balance
if opt.BalanceTrain
    % Randomly downsample the larger class so that faces and scenes have
    % equal numbers of trials in the training set.
    idx_f = find(train_mask & labels== 1);
    idx_s = find(train_mask & labels==-1);
    nmin  = min(numel(idx_f), numel(idx_s));
    keep  = false(size(train_mask));
    keep([randsample(idx_f,nmin); randsample(idx_s,nmin)]) = true;
    train_mask = keep;
end

% Report how many trials per condition will be used for training
print_cond_counts(conditions, train_mask, 'Training set after filters');
print_run_cond_counts(conditions, runs, train_mask, 'Training per run after filters');

% Ensure every requested training run has at least one trial
for rr = train_runs(:)'
    if sum(train_mask & runs==rr)==0
        error('mvpa_MR:noTrainTrials', ...
            'No training trials for %s in run %d after filtering', ...
            strjoin(string(train_labels), ', '), rr);
    end
end

if ~any(train_mask)
    error('mvpa_MR:noTrainTrials', ...
        'No training trials remain after filtering for %s in runs %s', ...
        strjoin(string(train_labels), ', '), mat2str(train_runs));
end

[cv_res, cv_cm] = run_cv_block(betaFiles, labels, runs, train_mask, cfg, opt);
% cv_res contains the full TDT output and cfg used for the CV step, while
% cv_cm is a convenient 2x2 confusion matrix summarising performance.

%% -------- XCLASS: Train on full training set and test per run --------
% Each row in xclass_specs describes a cross-classification test. We iterate
% over them, training on ALL trials selected above and testing on the
% specified runs for that label set.
xclass_out = struct();
for i = 1:size(xclass_specs,1)
    test_labels = string(xclass_specs{i,1});
    test_runs   = xclass_specs{i,2};
    tag         = xclass_specs{i,3};
    test_filter = struct();
    if size(xclass_specs,2) >= 4 && ~isempty(xclass_specs{i,4}) && isstruct(xclass_specs{i,4})
        test_filter = xclass_specs{i,4};
    end

    test_mask_all = ismember(conditions, test_labels) & ismember(runs, test_runs);
    if ~isempty(fieldnames(test_filter))
        keep_test = apply_filters(Tbl, test_filter);
        test_mask_all = test_mask_all & keep_test;
    end
    print_cond_counts(conditions, test_mask_all, sprintf('Test set %s after filters', tag));
    print_run_cond_counts(conditions, runs, test_mask_all, sprintf('Test per run after filters %s', tag));
    if ~any(test_mask_all)
        error('mvpa_MR:noTestTrials', ...
            'No test trials remain for %s (%s) in runs %s after filtering', ...
            tag, strjoin(test_labels, ', '), mat2str(test_runs));
    end
    xclass_out.(tag) = run_xclass_per_runs(betaFiles, labels, runs, train_mask, test_mask_all, test_runs, test_labels, cfg, tag, out_dir, opt);
end

%% -------- Summary table --------
% Combine cross-validation and cross-classification results into a single
% table with a variety of performance metrics.
% Compute extended metrics tables
summary_tbl = build_summary_table(cv_res, xclass_out);
% Write CSV
try
    writetable(summary_tbl, fullfile(out_dir,'summary_all.csv'));
catch
    warning('Could not write CSV');
end

%% -------- Pack results --------
results.cv      = cv_res;
results.cv.cm   = cv_cm;
results.xclass  = xclass_out;
results.table   = summary_tbl;

save(fullfile(out_dir,'summary_all.mat'),'results');

end % main

%% ====================== SUBFUNCTIONS ======================
% Helper utilities used by the main function. These are kept below to keep the
% main workflow easy to read.
function study_data = fix_paths_in_struct(study_data, old_root, new_root)
% Replace old directory prefixes inside the study_data struct. Useful when
% moving data between machines.
fields_ruta = {'beta_files','beta_file','beta','file','fname','path'};
for f = fields_ruta
    fn = f{1};
    if isfield(study_data, fn)
        for i = 1:numel(study_data)
            if ischar(study_data(i).(fn)) || isstring(study_data(i).(fn))
                study_data(i).(fn) = strrep(char(study_data(i).(fn)), old_root, new_root);
            end
        end
    end
end
end

function [cv_res, cm] = run_cv_block(betaFiles, labels, runs, tr_mask, cfg, opt)
% Helper that performs leave-one-run-out cross-validation within the
% specified training mask.
bF   = betaFiles(tr_mask);
lab  = labels(tr_mask);
chnk = runs(tr_mask); % use run as chunk

cfg.files.name  = bF;
cfg.files.label = lab;
cfg.files.chunk = chnk;

cfg.design = make_design_cv(cfg);
cfg.design.unbalanced_data   = 'ok';
cfg.scale.check_datatrans_ok = true;

res = decoding(cfg);
cm  = fetch_cm(res);

% binomial p-value for accuracy using TDT's decoding_statistics
stats_cfg = cfg;
stats_cfg.stats.test   = 'binomial';
stats_cfg.stats.tail   = 'right';
stats_cfg.stats.output = 'accuracy_minus_chance';
stats_cfg.stats.chancelevel = 50; % percent
acc_p = decoding_statistics(stats_cfg, res);

if opt.SavePNGs
    fig = figure('Visible','off'); heatmap(cm,'Colormap',jet);
    title('CV confusion matrix');
    saveas(fig, fullfile(cfg.results.dir, 'CV_confusion.png')); close(fig);
end

cv_res.results = res;
cv_res.cfg     = cfg;
cv_res.cm      = cm;
cv_res.acc_p   = acc_p;
end

function outStruct = run_xclass_per_runs(betaFiles, labels, runs, tr_mask, te_mask_all, run_list, test_labels, cfg, tag, out_dir, opt)
% Train on all trials in tr_mask and test separately on each run in
% run_list for the given set of labels. Results are averaged across runs.
outStruct = struct();
cm_list  = cell(numel(run_list),1);
vec_list = cell(numel(run_list),1);
acc_list = cell(numel(run_list),1);
auc_list = cell(numel(run_list),1);
p_list = cell(numel(run_list),1);

idx_train = find(tr_mask);

% Loop over runs of interest. For each run we build a TDT design that uses
% all training trials and the trials of the current run as the test set.

for k = 1:numel(run_list)
    r = run_list(k);
    idx_test = find(te_mask_all & runs==r);
    if isempty(idx_test)
        warning('No test trials for %s (%s) in run %d', tag, strjoin(test_labels, ', '), r);
        cm = nan(2); vec = cm(:)';
    else
        keep_idx = unique([idx_train; idx_test]);
        names = betaFiles(keep_idx);
        labs  = labels(keep_idx);

        setvec = ones(numel(keep_idx),1); % 1=train
        [~,loc_test] = ismember(idx_test, keep_idx);
        setvec(loc_test) = 2;             % 2=test
        chunk  = (1:numel(keep_idx))';

        cfg.files.name   = names;
        cfg.files.label  = labs;
        cfg.files.chunk  = chunk;
        cfg.files.xclass = setvec;

        cfg.design = make_design_xclass(cfg);
        cfg.design.unbalanced_data   = 'ok';
        cfg.scale.check_datatrans_ok = true;
        if opt.Overwrite, cfg.results.overwrite = 1; end

        res = decoding(cfg);
        cm  = fetch_cm(res);
        vec = cm(:)';

        % p-value using decoding_statistics for this run
        stats_cfg = cfg;
        stats_cfg.stats.test   = 'binomial';
        stats_cfg.stats.tail   = 'right';
        stats_cfg.stats.output = 'accuracy_minus_chance';
        stats_cfg.stats.chancelevel = 50; % percent
        p_run = decoding_statistics(stats_cfg, res);

        acc_mc = getfield_safe(res,'accuracy_minus_chance',NaN);
        auc_mc = getfield_safe(res,'AUC_minus_chance',NaN);

        save(fullfile(out_dir, sprintf('%s_run%02d_%s.mat', tag, r, 'xclass')), 'res','cfg');
    if opt.SavePNGs
        fig = figure('Visible','off'); heatmap(cm,'Colormap',jet);
        title(sprintf('%s – run %d', tag, r));
        saveas(fig, fullfile(out_dir, sprintf('%s_run%02d_conf.png', tag, r))); close(fig);
    end
    end
    cm_list{k}  = cm;
    vec_list{k} = vec;
    acc_list{k} = acc_mc;
    auc_list{k} = auc_mc;
    p_list{k}   = p_run;
end

% Aggregate results across runs
cm_stack  = cat(3, cm_list{:});
mean_cm   = mean(cm_stack,3,'omitnan');
vec_stack = cell2mat(vec_list');
mean_vec  = mean(vec_stack,1,'omitnan');
acc_mean  = mean(cell2mat(acc_list), 'omitnan');
auc_mean  = mean(cell2mat(auc_list), 'omitnan');

cm_total = sum(cm_stack,3,'omitnan');
% Non‑integer values can appear when averaging across runs. Round before
% computing significance so that stats_binomial receives integers.
n_correct = round(sum(diag(cm_total)));
n_total   = round(sum(cm_total(:)));
acc_p = stats_binomial(n_correct,n_total,0.5,'right');

% Plot & save mean CM for this tag
if opt.SavePNGs
    fig = figure('Visible','off');
    heatmap(mean_cm,'Colormap',jet);
    title(sprintf('%s – mean confusion matrix', tag));
    saveas(fig, fullfile(out_dir, sprintf('%s_mean_conf.png', tag)));
    close(fig);
end

outStruct.cm_list  = cm_list;
outStruct.vec_list = vec_list;
outStruct.mean_cm  = mean_cm;
outStruct.mean_vec = mean_vec;
outStruct.acc_list = acc_list;
outStruct.acc_mean = acc_mean;
outStruct.auc_list = auc_list;
outStruct.auc_mean = auc_mean;
outStruct.p_list   = p_list;
outStruct.acc_p    = acc_p;
end

function tbl = build_summary_table(cv_res, xclass_out)
% Build a summary table with multiple metrics

% ---- CV metrics ----
cm_cv = cv_res.cm; [m_cv] = derive_metrics_from_cm(cm_cv);
acc_cv  = cv_res.results.accuracy_minus_chance.output;
auc_cv  = getfield_safe(cv_res.results,'AUC_minus_chance',NaN); %#ok<GFLD>
acc_p_cv = cv_res.acc_p;

RowID = {'CV'}; Type = {'cv'}; Tag = {'cv'};
AccMinusChance_TDT = acc_cv;   % direct from TDT
BalancedAcc        = m_cv.balanced_acc;
Sensitivity    = m_cv.sensitivity;
Specificity    = m_cv.specificity;
MCC            = m_cv.mcc;
Kappa          = m_cv.kappa;
AUCminusChance = auc_cv;
AccPValue = acc_p_cv;

% ---- XCLASS metrics ----
fn = fieldnames(xclass_out);
for i = 1:numel(fn)
    tag = fn{i};
    mean_cm = xclass_out.(tag).mean_cm;
    m = derive_metrics_from_cm(mean_cm);
    RowID{end+1,1} = ['XCLASS_' tag]; %#ok<AGROW>
    Type{end+1,1}  = 'xclass';
    Tag{end+1,1}   = tag;
    AccMinusChance_TDT(end+1,1) = xclass_out.(tag).acc_mean;
    BalancedAcc(end+1,1)    = m.balanced_acc;
    Sensitivity(end+1,1)    = m.sensitivity;
    Specificity(end+1,1)    = m.specificity;
    MCC(end+1,1)            = m.mcc;
    Kappa(end+1,1)          = m.kappa;
    AUCminusChance(end+1,1) = xclass_out.(tag).auc_mean;
    AccPValue(end+1,1)      = xclass_out.(tag).acc_p;
end

tbl = table(RowID,Type,Tag,AccMinusChance_TDT,BalancedAcc,Sensitivity,Specificity,MCC,Kappa,AUCminusChance,AccPValue, ...
             'VariableNames',{'RowID','Type','Tag','AccuracyMinusChance_TDT','BalancedAcc','Sensitivity','Specificity','MCC','Kappa','AUCminusChance','AccPValue'});
end

function m = derive_metrics_from_cm(cm)
% Utility that computes common binary classification metrics from a 2x2
% confusion matrix. Returns NaNs if the matrix is invalid.
m = struct('sensitivity',NaN,'specificity',NaN,'balanced_acc',NaN,'mcc',NaN,'kappa',NaN);
if ~all(size(cm)==[2 2]) || any(isnan(cm(:)))
    return;
end
TP = cm(1,1); FN = cm(1,2); FP = cm(2,1); TN = cm(2,2);
P = TP+FN; N = TN+FP; total = P+N;
if P>0, m.sensitivity = TP/P; end
if N>0, m.specificity = TN/N; end
if ~isnan(m.sensitivity) && ~isnan(m.specificity)
    m.balanced_acc = (m.sensitivity + m.specificity)/2;
end
num = TP*TN - FP*FN;
den = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
if den>0, m.mcc = num/den; end
po = (TP+TN)/total;
pe = ((TP+FP)*(TP+FN) + (FN+TN)*(FP+TN)) / (total^2);
if (1-pe)~=0, m.kappa = (po - pe)/(1-pe); end
end

function val = getfield_safe(S, fieldname, default)
% Helper to read optional fields from the TDT results structure.
if isfield(S, fieldname)
    val = S.(fieldname).output;
else
    val = default;
end
end

function cm = fetch_cm(res)
% Extract the 2x2 confusion matrix from TDT results structures. Some
% configurations produce a *_plus_undecided matrix; this helper deals with
% both variants and trims any extra rows/columns.
if isfield(res,'confusion_matrix_plus_undecided')
    C = res.confusion_matrix_plus_undecided.output;
elseif isfield(res,'confusion_matrix')
    C = res.confusion_matrix.output;
else
    error('No confusion matrix field found in results struct.');
end
if iscell(C), C = C{1}; end
% If there is an undecided column, drop it
if size(C,2) > 2
    C = C(:,1:2);
end
if size(C,1) > 2
    C = C(1:2,:);
end
cm = C;
end

function keep = apply_filters(Tsub, filt)
% Apply field-based filters to a table of trials (Tsub)
% filt: struct; each field is a column name in Tsub with value:
%   - cellstr/string array of allowed values
%   - numeric array of allowed numbers
%   - function handle @(col) -> logical vector
keep = true(height(Tsub),1);
fn = fieldnames(filt);
for ii = 1:numel(fn)
    f = fn{ii};
    if ~ismember(f, Tsub.Properties.VariableNames)
        warning('apply_filters: field "%s" not found. Ignored.', f);
        continue;
    end
    col = Tsub.(f);
    val = filt.(f);
    if isa(val,'function_handle')
        k = val(col);
        if ~islogical(k) || numel(k)~=height(Tsub)
            error('Filter function for %s must return logical vector of length %d', f, height(Tsub));
        end
        keep = keep & k;
    elseif iscellstr(val) || isstring(val)
        keep = keep & ismember(string(col), string(val));
    elseif isnumeric(val)
        keep = keep & ismember(col, val);
    else
        error('Unsupported filter type for field %s', f);
    end
end

end

function print_cond_counts(cond_names, mask, header)
% Utility to print number of trials per condition after applying a mask
if nargin < 3, header = ''; end
if ~isempty(header)
    fprintf('%s\n', header);
end
u = unique(cond_names(mask));
for i = 1:numel(u)
    fprintf('Trials for %s: %d\n', u(i), sum(cond_names(mask)==u(i)));
end
end

function print_run_cond_counts(cond_names, runs, mask, header)
% Print number of trials per run for each condition given a mask
if nargin < 4, header = ''; end
if ~isempty(header)
    fprintf('%s\n', header);
end
u_runs  = unique(runs(mask));
u_cond  = unique(cond_names(mask));
for r = u_runs(:)'
    fprintf('Run %d:\n', r);
    for c = 1:numel(u_cond)
        cnt = sum(mask & runs==r & cond_names==u_cond(c));
        fprintf('  %s: %d\n', u_cond(c), cnt);
    end
end
end

