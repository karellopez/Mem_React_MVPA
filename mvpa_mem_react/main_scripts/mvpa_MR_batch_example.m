
clear; clc; close all

%% Inputs

% Matfiles
    mat_files = { ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/MFA_000_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_001_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_002_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_003_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_004_study_data.mat', ...
        '/home/karelo/Desktop/Development/scripts_Sabina/AFNI_analysis/5.trial_wise_mat_files/yseg_005_study_data.mat', ...
        % add more files here
        };
 % Output dir
    output_dir = '/home/karelo/Desktop/Development/MVPA_Merchi/resutls_mvpa';
    
 % ROI  
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
group_table = table();

delete(gcp("nocreate"));
parpool("Processes")
parfor i = 1:numel(mat_files)

    % mat files
    study_mat = mat_files{i};
    % Sub name (adapta esta parte)
    [path,sub] = fileparts(study_mat);
    sub_name = strrep (sub,'_study_data','');
    % Sub output folder
    sub_out_dir = fullfile(output_dir, sub_name);if ~exist(sub_out_dir, 'dir'); mkdir(sub_out_dir); end

        res = mvpa_MR( ...
            study_mat, mask_file, sub_out_dir, ...
            train_labels, train_runs, xclass_specs, ...
            'BalanceTrain', false, ...
            'SavePNGs',     true, ...
            'Overwrite',    true, ...
            'TrainFilter', TrainFilter, ...
            'FixOldPath',   {'/home/karelo/Escritorio/','/home/karelo/Desktop/'} ); % Esta linea de 'FixOldPath'no la necesitas. Es solo un apaño para reemplazar strings en el study mat file cuando los paths eran de otro ordenador

        % Append the final table for this subject to the group table
        final_tbl = res.final;
        final_tbl.Subject = string(sub_name);
        % Place the Subject column first in the table
        final_tbl = movevars(final_tbl, 'Subject', 'Before', 1);
        group_table = [group_table; final_tbl];

         close all

end

% Save group-level summary table with one row per subject
writetable(group_table, fullfile(output_dir, 'group_summary.csv'));
save(fullfile(output_dir, 'group_summary.mat'), 'group_table', 'cm_means');
