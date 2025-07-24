% This is an example how to create a design from a table, e.g. a csv/tsv/txt file
%
% Kai, 2021/06/09

cfg = decoding_defaults;
designfile = 'demo13_design_from_csv_example.txt'

%% step 1: load the description as table
% load data as table
designtable = readtable(designfile) % , 'UseExcel', false)

% this will result in a table like
% designtable =
% 
%   4×3 table
% 
%        file        condition    chunk
%     ___________    _________    _____
% 
%     {'f1a.nii'}     {'l1'}        1  
%     {'f1b.nii'}     {'l1'}        2  
%     {'f2a.nii'}     {'l2'}        1  
%     {'f2b.nii'}     {'l2'}        2  

% note: the condition is so far an arbitrary string
%       they are converted to labels next

%% step 2.1: convert to the TDT-specific "regressor_names" variables
% (see e.g. make_design_from_spm for a description)

% if your table column names differ from the example, adapt the code below.
% also make sure that only the chunk variable is numeric, and the rest is
% a cellstr (e.g. using num2str)

% the "regressor_names" variable has the following format + content
%   regressor_names: a 3-by-n cell matrix.

%   regressor_names(1, :) -  shortened names of the regressors
%     If more than one basis function exists in the design matrix (e.g. as is
%     the case for FIR designs), each regressor name will be extended by a
%     string ' bin 1' to ' bin m' where m refers to the number of basis
%     functions.
regressor_names(1, :) = designtable.condition;

%   regressor_names(2, :) - experimental run/session of each regressors
regressor_names(2, :) = num2cell(designtable.chunk);

%   regressor_names(3, :) - some optional information, here we put the
%       filename to verify beta_loc later
regressor_names(3, :) = designtable.file

%% step 2.2: add the TDT-specific "beta_loc" variables
% the "beta_loc" variable traditionally only contained the folder in which 
% a SPM.mat resided and where it's corresponding beta images were placed.
% now it can also contain the names of the input files as a cellstr.
% note: the file names must be in the same order as in "regressor_names" 
beta_loc = regressor_names(3, :)

%% step 3: good to go. work as usual

% assign labels for the condition strings
labelnames = {'l1', 'l2'};
labels = [1, 2];

% describe the data
cfg = decoding_describe_data(cfg,labelnames, labels,regressor_names,beta_loc);

% create a cv-design
cfg.design = make_design_cv(cfg);
plot_design(cfg);

% dont write output in this example
disp('Setting cfg.results.write = 0 to avoid writing output in this example')
cfg.results.write = 0;

% note: if you do not use SPM or AFNI file formats, make sure to have a
% suitable read function for the files and set it as cfg.software, 
% or see others "help decoding" and other demos on how to use the 
% "passed_data" variable and preload the data here

%% enjoy (if files would exist that are called f1a.nii etc)
[results, cfg] = decoding(cfg);