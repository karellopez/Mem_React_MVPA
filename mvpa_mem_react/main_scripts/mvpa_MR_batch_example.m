% Example script to run mvpa_MR on multiple subjects
clc; clear; close all

% List of subject study_data files
mat_files = { ...
    'subj01_study_data.mat', ...
    'subj02_study_data.mat'  ...
    % add more files here
    };

mask_file = 'path/to/roi_mask.nii';
train_labels = {'DE_faces','DE_scenes'};
train_runs   = 1:4;

% Cross-classification specifications
xclass_specs = { {'AB_faces','AB_scenes'}, 1:4, 'AB' };

for i = 1:numel(mat_files)
    study_mat = mat_files{i};
    out_dir = fullfile('results', sprintf('subj%02d', i));
    if ~exist(out_dir, 'dir'); mkdir(out_dir); end
    mvpa_MR(study_mat, mask_file, out_dir, ...
            train_labels, train_runs, xclass_specs);
end

