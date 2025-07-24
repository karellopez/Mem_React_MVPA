% This script is a demo showing 
%   1. that libsvm handles assigns samples for which no clear class can be 
%      predicted to label 1, and 
% 2. that this can be visualised with 
%           cfg.results.output = {'confusion_matrix_plus_undecided'};
%       instead of 
%           cfg.results.output = {'confusion_matrix'};
% The reason why libsvm does this is that it is forced to do a decision,
% and deciding for the first label is a) reproducibale and b) does not 
% provide a systematic error for accuracy (but of course one for the
% predictions). Look at the libsvm website for more on how libsvm handles 
% multiclass cases (in short: they apply pairwise comparisons and then use
% the majority vote).
%
% You might need to run this function multiple times to see the effect (the
% lower figure will have the number of the left column of the upper figure
% as a combination of left column and 'undecided')

% Kai 2021/03/16

fpath = fileparts(fileparts(mfilename('fullpath')));
addpath(fpath)

clear variables
dbstop if error % if something goes wrong


nrep = 1 % choose this to visualize 1 realisation
% nrep = 1000 % uncomment if you like to get mean and ci95 for many
              % realisations

for rep_ind = 1:nrep
    rep_ind
    %% Set the output directory where data will be saved
    cfg = decoding_defaults;
    if nrep > 1, cfg.plot_design = 0; end
    % cfg.results.dir = % e.g. 'toyresults'
    cfg.results.write = 0; % no results are written to disk
    
    %% generate some toy data
    % define number of "runs" and center means
    nruns = 4; % lets simulate we have n runs
    ntrial_per_run = 20;
    ntrial_tot = nruns * ntrial_per_run;
    
%     demomode = 'samemeans'; % alternative 1: same means for all data: that should show a bias for the first class for libsvm
    demomode = 'diffmeans'; % alternative 2: that probably shows only
    % very little bias, as there are hardly any ties (unclear classifications)
    switch demomode
        case 'diffmeans' % MAKE SURE TO SET demomode = 'diffmeans' ABOVE to use different means
            set1.mean = [0 0];
            set2.mean = [1 1]; % should have the same dim as set1, otherwise it wont work (and would not make sense, either)
            set3.mean = [-1 1];
            set4.mean = [-1 -1];
        case 'samemeans'
            set1.mean = [0 0];
            set2.mean = [0 0]; % should have the same dim as set1, otherwise it wont work (and would not make sense, either)
            set3.mean = [0 0];
            set4.mean = [0 0]
    end
    
    % add gaussian noise around the mean
    data1 = 1 * randn(ntrial_tot, length(set1.mean)) + repmat(set1.mean, ntrial_tot, 1);
    data2 = 1 * randn(ntrial_tot, length(set2.mean)) + repmat(set2.mean, ntrial_tot, 1);
    data3 = 1 * randn(ntrial_tot, length(set3.mean)) + repmat(set3.mean, ntrial_tot, 1);
    data4 = 1 * randn(ntrial_tot, length(set4.mean)) + repmat(set4.mean, ntrial_tot, 1);
    
    % put all together in a data matrix
    data = [data1; data2; data3; data4];
    
    %% add data description
    % save labels
    cfg.files.label = [-2*ones(size(data1,1), 1); 7*ones(size(data2,1), 1); 5*ones(size(data3,1), 1); 4*ones(size(data4,1), 1)];
    
    % save "run" as chunk number
    cfg.files.chunk = [
        sort(repmat(1:nruns, 1, ntrial_per_run)), ... % class 1
        sort(repmat(1:nruns, 1, ntrial_per_run)), ... % class 2
        sort(repmat(1:nruns, 1, ntrial_per_run)), ... % class 3
        sort(repmat(1:nruns, 1, ntrial_per_run)), ... % class 4
        ]';
    % join run 1 2 to run 1, run 3 4 to run 2, etc, to get two sample per run
    % per class
    cfg.files.chunk = ceil(cfg.files.chunk/2);
    
    all_chunks = unique(cfg.files.chunk);
    all_labels = unique(cfg.files.label);
    
    % save a description
    ct = zeros(length(all_labels),length(all_chunks));
    for ifile = 1:length(cfg.files.label)
        curr_label = cfg.files.label(ifile);
        curr_chunk = cfg.files.chunk(ifile);
        f1 = all_labels==curr_label; f2 = all_chunks==curr_chunk;
        ct(f1,f2) = ct(f1,f2)+1;
        cfg.files.name(ifile) = {sprintf('class%i_run%i_%i', curr_label, curr_chunk, ct(f1,f2))};
    end
    
    % add an empty mask
    cfg.files.mask = '';
    
    %% plot the data (if 2d)
    if size(data, 2) == 2
        try
            figure(resfig);
        catch
            resfig = figure('name', 'Data');
        end
        scatter(data(:, 1), data(:, 2), 30, cfg.files.label);
    end
    
    %% Prepare data for passing
    passed_data.data = data;
    passed_data.mask_index = 1:size(data, 2); % use all voxels
    passed_data.files = cfg.files;
    passed_data.hdr = ''; % we don't need a header, because we don't write img-files as output (but mat-files)
    passed_data.dim = [length(set1.mean), 1, 1]; % add dimension information of the original data
    % passed_data.voxelsize = [1 1 1];
    
    
    %% Add defaults for the remaining parameters that we did not specify
    
    % Set the analysis that should be performed (here we only want to do 1
    % decoding)
    cfg.analysis = 'wholebrain';
    cfg.results.output = {'confusion_matrix', 'confusion_matrix_plus_undecided'};
    
    %% Nothing needs to be changed below for a standard leave-one-run out cross validation analysis.
    % Create a leave-one-run-out cross validation design:
    % cfg.design = make_design_cv(cfg);
    
    cfg.design = make_design_cv(cfg);
    
    % plot_design(cfg);
    
    %% Decoding Parameters
    
    % default: -s 0 -t 0 -c 1 -b 0 -q
    cfg.decoding.method = 'classification'; % the calculation will also work with _kernel methods, but plotting of the classification surface below is only implemented for the non-kernel method
    cfg.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 0 -q';
    cfg.scale.method = 'min0max1';
    cfg.scale.estimation = 'all'; % scaling across all data is equivalent to no scaling (i.e. will yield the same results), it only changes the data range which allows libsvm to compute faster
    
    %% Run decoding
    [results, cfg] = decoding(cfg, passed_data);
    coll1(:, :, rep_ind) = results.confusion_matrix.output{1};
    coll2(:, :, rep_ind) = results.confusion_matrix_plus_undecided.output{1};

end

%% Show statistics if more than 1 repetition
if nrep > 1
    m1 = mean(coll1, 3);
    m2 = mean(coll2, 3);

    ci95_1 = 1.96 * std(coll1, '', 3) / sqrt(nrep);
    ci95_2 = 1.96 * std(coll2, '', 3) / sqrt(nrep);

    disp('Showing mean (first nxn entries) and CI95 (last nxn entries + nx1 undecided) for (1) confusion matrix and (2) confusion_matrix_plus_undecided ')
    mean1_ci95_1 = [m1, ci95_1]
    mean2_ci95_2 = [m2, ci95_2]

    % EXAMPLE result from this with nrep = 1000:
    %
    % mean1_ci95_1 =
    %    Class1    Class2    Class3     CI95_1    CI95_2    CI95_3
    %    35.1100   32.7275   32.1625    0.7280    0.6815    0.6908
    %    35.3813   32.8075   31.8113    0.7075    0.7251    0.6967
    %    35.2050   32.8950   31.9000    0.7009    0.6939    0.6942
    % 
    % 
    % mean2_ci95_2 =
    %    Class1    Class2    Class3     Undec     CI95_1    CI95_2    CI95_3    CI95_Undec
    %    32.7488   32.7275   32.1625    2.3613    0.7184    0.6815    0.6908    0.1766
    %    32.9838   32.8075   31.8113    2.3975    0.6976    0.7251    0.6967    0.1665
    %    32.9387   32.8950   31.9000    2.2662    0.6845    0.6939    0.6942    0.1735
    % 
    % MEANING for (1) confusion matrix: about ~35.x % +-0.7x % CI95 fall in
    %   the first class, in comparison to ~32.x % +-0.7x in class 2 and 3
    %   (sig. difference and bias from about 2% for cass 1)
    %
    % which is shown in:
    % 
    % MEANING for (2) confusion matrix plus undecided:
    %  about ~32.x % +-0.7x % CI95 fall in classes 1-3 (first three columns)
    %  (no sig. difference)
    %    and ~2.3x % +-0.17x % CI95 are undecided
    %   
    % NOTE that the amount of undecided classifications depends on many 
    % things, including your signal and noise strength and type of your 
    % data, the amount of classes in your multiclass, etc
    
%% Show original confusion matrix and confusion matrix with undecided cases
else
    figure('name', 'demo3_1:confusion_matrix vs confusion_matrix_plus_undecided', 'Position', [27 407 890 573])
    subplot(2,2,1)
    confmat = results.confusion_matrix.output{1};
    imagesc(confmat)
    axis tight
    axis equal
    title('confusion_matrix', 'Interpreter', 'none')
    colorbar;
    set(gca, 'XTick', 1:length(all_labels), 'XTickLabel', all_labels)
    set(gca, 'YTick', 1:length(all_labels), 'YTickLabel', all_labels)
    for rowpos = 1:size(confmat, 1)
        for colpos = 1:size(confmat, 2)
            text(colpos, rowpos, num2str(confmat(rowpos, colpos)), 'HorizontalAlignment', 'center');
        end
    end

    subplot(2,2,3)
    confmat = results.confusion_matrix_plus_undecided.output{1};
    imagesc(confmat)
    axis tight
    axis equal
    title('confusion_matrix_plus_undecided', 'Interpreter', 'none')
    colorbar;
    set(gca, 'XTick', 1:length(all_labels)+1, 'XTickLabel', {num2str(all_labels), 'undecided'})
    set(gca, 'YTick', 1:length(all_labels), 'YTickLabel', all_labels)
    for rowpos = 1:size(confmat, 1)
        for colpos = 1:size(confmat, 2)
            text(colpos, rowpos, num2str(confmat(rowpos, colpos)), 'HorizontalAlignment', 'center');
        end
    end

    subplot(2,2,2)
    infotext = {
        ['Current demomode: ' demomode]
     'demomode samemeans means: no difference between classes';
     'demomode diffmeans means: some differnce between the classes';
     'you can change the mode and the means in the demo script';
     ' ';
     'About this script:';
     'This script is a demo showing';
    '  1. that libsvm handles assigns samples for which no clear class can be';
    '     predicted to label 1, and';
    '2. that this can be visualised with';
    '          cfg.results.output = {''confusion_matrix_plus_undecided''};';
    '      instead of';
    '          cfg.results.output = {''confusion_matrix''};';
    'The reason why libsvm does this is that it is forced to do a decision,';
    'and deciding for the first label is a) reproducibale and b) does not';
    'provide a systematic error for accuracy (but of course one for the';
    'predictions). Look at the libsvm website for more on how libsvm handles';
    'multiclass cases (in short: they apply pairwise comparisons and then use';
    'the majority vote).';
    ' ';
    'You might need to run this function multiple times to see the effect (the';
    'lower figure will have the number of the left column of the upper figure';
    'as a combination of left column and ''undecided'')';
    }
    text(0, 0, infotext, 'Interpreter', 'None', 'FontSize', 8);
    axis off
end