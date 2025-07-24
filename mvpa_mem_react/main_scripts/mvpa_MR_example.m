clc;clear;close all
% --- Required ---------------------------------------------------------
study_mat   = '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_003_study_data.mat';
mask_file   = '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/ROIs/mvpa_mask_2_res.nii';
out_dir     = '/home/karelo/Desktop/Development/MVPA_Merchi/results/';

train_labels = {'DE_faces','DE_scenes'};   % CV training set
train_runs   = 1:4;                        % runs used in CV

% Cross-classification specs: {test_labels, test_runs, tag}
xclass_specs = {
    {'AB_faces','AB_scenes'}, 1:4, 'AB';
    {'AC_faces','AC_scenes'}, 2:5, 'AC'
};

% --- Optional name/value pairs ---------------------------------------
results = mvpa_MR( ...
    study_mat, mask_file, out_dir, ...
    train_labels, train_runs, xclass_specs, ...
    'BalanceTrain', false, ...
    'SavePNGs',     true, ...
    'Overwrite',    true, ...
    'FixOldPath',   {'/home/karelo/Escritorio/','/home/karelo/Desktop/'} );


clc;clear;close all
% --- Required ---------------------------------------------------------
study_mat   = '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_003_study_data.mat';
mask_file   = '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/ROIs/mvpa_mask_2_res.nii';
out_dir     = '/home/karelo/Desktop/Development/MVPA_Merchi/results2/';

train_labels = {'DE_faces','DE_scenes'};   % CV training set
train_runs   = 1:4;                        % runs used in CV

% Cross-classification specs: {test_labels, test_runs, tag}
% Filtros para CV (opcional)
TrainFilter = struct('source_encoding',{{'hit'}}, ...
                     'item_recognition',{{'hit'}});
xclass_specs = {
    {'AB_faces','AB_scenes'}, 1:4, 'AB', struct('source_encoding',{{'hit'}}, 'item_recognition',{{'hit'}})
    {'AC_faces','AC_scenes'}, 2:5, 'AC', struct('source_encoding',{{'hit'}}, 'item_recognition',{{'hit'}})
};


% --- Optional name/value pairs ---------------------------------------
results = mvpa_MR( ...
    study_mat, mask_file, out_dir, ...
    train_labels, train_runs, xclass_specs, ...
    'BalanceTrain', false, ...
    'SavePNGs',     true, ...
    'Overwrite',    true, ...
    'FixOldPath',   {'/home/karelo/Escritorio/','/home/karelo/Desktop/'} );