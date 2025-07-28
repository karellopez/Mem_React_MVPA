
clear; clc; close all

%% List of subject files and ROI mask
    mat_files = { ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/MFA_000_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_001_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_002_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_003_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_004_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_005_study_data.mat', ...
        % add more files here
        };
    
  %% PROCESS
    mask_file   = '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/ROIs/mvpa_mask_2_res.nii';

 % CROSS-VALIDATION (CV) leave-one-run-out
    % Labels for initial CV model and training set in further XCLASS model
    train_labels = {'DE_faces','DE_scenes'};
    % Filter trials by performance
    TrainFilter = struct('item_recognition',{{'hit'}});
    % runs to include in the model
    train_runs   = 1:5;
    
    % XCLASS model, labels and filters
        % Here, we set the conditions in which we want to test the model
        % (one condition per raw). You can also filter trials by
        % performance in cognitive task. 
    
    xclass_specs = {
        {'AB_faces','AB_scenes'}, 1:4, 'AB', struct('item_recognition',{{'hit'}})
        {'AC_faces','AC_scenes'}, 2:5, 'AC', struct('item_recognition',{{'hit'}})
    };

%% Process each subject and store mean confusion matrices
cm_means = struct();
for i = 1:numel(mat_files)
    study_mat = mat_files{i};
    out_dir   = fullfile('results', sprintf('subj%02d', i));
        if ~exist(out_dir, 'dir'); mkdir(out_dir); end

        res = mvpa_MR( ...
            study_mat, mask_file, out_dir, ...
            train_labels, train_runs, xclass_specs, ...
            'BalanceTrain', false, ...
            'SavePNGs',     true, ...
            'Overwrite',    true, ...
            'TrainFilter', TrainFilter, ...
            'FixOldPath',   {'/home/karelo/Escritorio/','/home/karelo/Desktop/'} );
    
         close all

    cm_means(i).subject = i;
    cm_means(i).cv  = res.cv.cm;
    tags = fieldnames(res.xclass);
    for t = 1:numel(tags)
        tag = tags{t};
        cm_means(i).(tag) = res.xclass.(tag).mean_cm;
    end
 
end

save('confusion_means.mat','cm_means');