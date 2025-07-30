# Mem_React_MVPA

A small collection of Matlab utilities for multi-voxel pattern analysis (MVPA)
in a memory reactivation experiment. The entry point is `mvpa_MR.m` which wraps
the [decoding toolbox](https://github.com/mauricioferreira/decoding_toolbox) and
provides a reproducible pipeline for each subject.

## Features
- Loads `*_study_data.mat` files and optionally fixes old path roots.
- Builds a decoding toolbox configuration for ROI analyses.
- Leave-one-run-out cross‑validation on a user defined training set.
- Optional balancing of training trials and flexible trial filtering via the
  `TrainFilter` parameter.
- Cross‑classification on additional label/run splits.
- Outputs confusion matrices, PNG visualisations and a summary CSV with several
  performance metrics.
- Computes binomial p-values via TDT's `decoding_statistics`.

## Usage
```
TrainFilter = struct('source_encoding', {'hit'}, 'item_recognition', {'hit'});
xclass_specs = {
    {'AB_faces','AB_scenes'}, 1:4, 'AB', struct('source_encoding', {'hit'}, 'item_recognition', {'hit'})
};

results = mvpa_MR(study_mat, mask_file, out_dir, ...
                  train_labels, train_runs, xclass_specs, ...
                  'TrainFilter', TrainFilter, ...
                  'BalanceTrain', false);
```

Ensure that any trial filters are passed through the `TrainFilter` parameter so
that they are applied during cross‑validation.

## Batch processing example
To analyse several subjects you can loop over their `*_study_data.mat` files
and create one output directory per subject. Below is a minimal example using
MATLAB:

```matlab
mat_files = { 'subj01_study_data.mat', 'subj02_study_data.mat' };
mask_file  = 'path/to/roi_mask.nii';
train_labels = {'DE_faces','DE_scenes'};
train_runs   = 1:4;
xclass_specs = { {'AB_faces','AB_scenes'}, 1:4, 'AB' };

for i = 1:numel(mat_files)
    study_mat = mat_files{i};
    out_dir = fullfile('results', sprintf('subj%02d', i));
    mvpa_MR(study_mat, mask_file, out_dir, ...
            train_labels, train_runs, xclass_specs);
end
```

Each subject will have its results stored under `results/subjXX/`.
