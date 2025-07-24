% This demo shows how to use multitarget regression for a circular label (degrees on
% a circle). The toy data are Matlab matrices and no real fMRI or EEG data,
% i.e. results do not necessarily generalize to real data.
%
% The script creates simple toy data in the form of an identity matrix
% where rows are the number of simulated samples (n_sample, see SET
% PARAMETERS) and columns are simulated voxels. A number of degree-labels
% (between 0 and 360 degrees) equal to the number of samples are created.
% These labels are projected into the space -pi:pi and sine and cosine
% components are extracted, which serve as the multiple targets for the
% regression. Sine and cosine are individually predicted using support
% vector regression with a number of cross-validation folds equal to
% n_run (see SET PARAMETERS). The predicted degree-label is reconstructed
% from the predicted sine and cosine components.
%
% Simon Weber, 2021/03/17

rng('shuffle')
clear variables

% check if decoding.m is in path, otherwise abort
if isempty(which('decoding.m'))
    error('Please add TDT to the matlab path')
end

%% SET PARAMETERS FOR SIMULATED DATA

n_sample = 40;
n_run = 8;
n_file = n_sample*n_run;

noise_scaling = 0.1;

%% SIMULATE DATA

data = eye(n_sample);
data = repmat(data,1,1,n_run);
data = data + randn(size(data)).*noise_scaling;

%% SIMULATE LABELS

% create labels (degrees btw. 0 and 360, equally spaced)
label_deg = linspace(360/n_sample,360,n_sample);
label_deg = repmat(label_deg,1,n_run)';
% project to range -pi:pi
label_rad = deg2rad(label_deg)-pi;
% extract sin and cos components
label_sin = sin(label_rad);
label_cos = cos(label_rad);
% turn into cell array for multitarget regression
label = mat2cell([label_sin, label_cos],ones(n_file,1));

chunk = sort(repmat(1:n_run,1,n_sample))';

%% PROCESS DATA FOR passed_data

% rescale data to [0,1] to decrease computing time
d = rescale(data);

% turn 3D array data_decode into 2D array by first exchanging
% (permute) dimensions 2 and 3 and then reshaping to [n_files,
% n_vox]
d = permute(d,[1,3,2]);
d = reshape(d,n_file,[],1);

%% SET UP TDT

% set up cfg
cfg = decoding_defaults;
cfg.analysis = 'wholebrain';
cfg.decoding.method = 'regression';
cfg.decoding.train.classification.model_parameters = '-s 4 -t 2 -c 1 -n 0.5 -b 0 -q';
cfg.decoding.software = 'libsvm_multitarget';
cfg.results.output = {'predicted_labels_multitarget'};
cfg.plot_selected_voxels = 0;
cfg.plot_design = 1;
cfg.multitarget = 1;
cfg.results.overwrite = 1;
cfg.results.write = 0;

%% PERFORM DECODING

% fill passed_data
passed_data.data = d;

% create design and perform decoding
[passed_data,cfg] = fill_passed_data(passed_data,cfg,label,chunk);
cfg.design = make_design_cv(cfg);

% perform decoding
[results,cfg,passed_data] = decoding(cfg,passed_data);


%% RECONSTRUCT ORIGINAL LABEL

% extract individual predictions
y_sin = results.predicted_labels_multitarget.output.model{1}.predicted_labels;
y_cos = results.predicted_labels_multitarget.output.model{2}.predicted_labels;

% convert back into rad
reconstructed_label = atan2(y_sin, y_cos)+pi;
reconstructed_label = rad2deg(reconstructed_label);

%% PLOT RESULTS

figure;
subplot(2,2,1);
plot(label_deg);
subplot(2,2,2);
plot(label_sin);
hold on
plot(label_cos);
subplot(2,2,3)
plot(reconstructed_label);
subplot(2,2,4)
plot(y_sin);
hold on
plot(y_cos);


