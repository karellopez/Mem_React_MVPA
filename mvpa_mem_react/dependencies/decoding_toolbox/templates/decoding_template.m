% This script is a template that can be used for a decoding analysis on 
% brain image data. It is for people who have betas available from an 
% SPM.mat and want to automatically extract the relevant images used for
% classification, as well as corresponding labels and decoding chunk numbers
% (e.g. run numbers). If you don't have this available, then use
% decoding_template_nobetas.m

% Make sure the decoding toolbox and your favorite software (SPM or AFNI)
% are on the Matlab path (e.g. addpath('/home/decoding_toolbox') )
% TDT
addpath('$ADD FULL PATH TO TDT TOOLBOX AS STRING OR MAKE THIS LINE A COMMENT IF IT IS ALREADY$')
assert(~isempty(which('decoding_defaults.m', 'function')), 'TDT not found in path, please add')
% SPM/AFNI
addpath('$ADD FULL PATH TO SPM/AFNI (if you need them) AS STRING OR MAKE THIS LINE A COMMENT IF IT IS ALREADY$')
assert((~isempty(which('spm.m', 'function')) || ~isempty(which('BrikInfo.m', 'function'))) , 'Neither SPM nor AFNI found in path, please add (or remove this assert if you really dont need to read brain images)')

% Set defaults
cfg = decoding_defaults;

% Set the analysis that should be performed (default is 'searchlight')
cfg.analysis = 'searchlight'; % standard alternatives: 'wholebrain', 'ROI' (pass ROIs in cfg.files.mask, see below)
cfg.searchlight.radius = 3; % use searchlight of radius 3 (by default in voxels), see more details below

% Set the output directory where data will be saved, e.g. 'c:\exp\results\buttonpress'
cfg.results.dir = 

% Set the filepath where your SPM.mat and all related betas are, e.g. 'c:\exp\glm\model_button'
beta_loc = 

% Set the filename of your brain mask (or your ROI masks as cell matrix) 
% for searchlight or wholebrain e.g. 'c:\exp\glm\model_button\mask.img' OR 
% for ROI e.g. {'c:\exp\roi\roimaskleft.img', 'c:\exp\roi\roimaskright.img'}
% You can also use a mask file with multiple masks inside that are
% separated by different integer values (a "multi-mask")
cfg.files.mask = 

% Set the label names to the regressor names which you want to use for 
% decoding, e.g. 'button left' and 'button right'
% don't remember the names? -> run display_regressor_names(beta_loc)
% infos on '*' (wildcard) or regexp -> help decoding_describe_data
labelname1  = 
labelname2  = 

labelvalue1 = 1; % value for labelname1
labelvalue2 = -1; % value for labelname2

%% Set additional parameters
% Set additional parameters manually if you want (see decoding.m or
% decoding_defaults.m). Below some example parameters that you might want 
% to use a searchlight with radius 12 mm that is spherical:

% cfg.searchlight.unit = 'mm';
% cfg.searchlight.radius = 12; % if you use this, delete the other searchlight radius row at the top!
% cfg.searchlight.spherical = 1;
% cfg.verbose = 2; % you want all information to be printed on screen
% cfg.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 0 -q'; 

% Enable scaling min0max1 (otherwise libsvm can get VERY slow)
% if you dont need model parameters, and if you use libsvm, use:
cfg.scale.method = 'min0max1';
cfg.scale.estimation = 'all'; % scaling across all data is equivalent to no scaling (i.e. will yield the same results), it only changes the data range which allows libsvm to compute faster

% if you like to change the decoding software (default: libsvm):
% cfg.decoding.software = 'liblinear'; % for more, see decoding_toolbox\decoding_software\. 
% Note: cfg.decoding.software and cfg.software are easy to confuse.
% cfg.decoding.software contains the decoding software (standard: libsvm)
% cfg.software contains the data reading software (standard: SPM/AFNI)

% Some other cool stuff
% Check out 
%   combine_designs(cfg, cfg2)
% if you like to combine multiple designs in one cfg.

%% Decide whether you want to see the searchlight/ROI/... during decoding
cfg.plot_selected_voxels = 500; % 0: no plotting, 1: every step, 2: every second step, 100: every hundredth step...

%% Add additional output measures if you like
% See help decoding_transform_results for possible measures

% cfg.results.output = {'accuracy_minus_chance', 'AUC'}; % 'accuracy_minus_chance' by default

% You can also use all methods that start with "transres_", e.g. use
%   cfg.results.output = {'SVM_pattern'};
% will use the function transres_SVM_pattern.m to get the pattern from 
% linear svm weights (see Haufe et al, 2015, Neuroimage)

%% Nothing needs to be changed below for a standard leave-one-run out cross
%% validation analysis.

% The following function extracts all beta names and corresponding run
% numbers from the SPM.mat
regressor_names = design_from_spm(beta_loc);

% Extract all information for the cfg.files structure (labels will be [1 -1] if not changed above)
cfg = decoding_describe_data(cfg,{labelname1 labelname2},[labelvalue1 labelvalue2],regressor_names,beta_loc);

% This creates the leave-one-run-out cross validation design:
cfg.design = make_design_cv(cfg); 

% Run decoding
results = decoding(cfg);