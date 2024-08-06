# Machine Learning of Brain-specific Biomarkers from EEG

This repository contains the research code from [Bomatter et al 2024, eBioMedicine](https://doi.org/10.1016/j.ebiom.2024.105259).

For the software library please checkout the [meeglet](https://roche.github.io/neuro-meeglet/) package.

## Install Instructions

1. Create a conda environment from the `environment.yml` file: `conda env create -f environment.yml`
2. Activate the environment: `conda activate eeg-biomaker-paper`
3. Install the core package: `pip install -e .`

## Configuration

1. Create a copy of `config.example.yml` and rename it to  `config.yml`
2. Adjust the configurations to specify the paths to the datasets in BIDS format as well as output paths where results should be saved.

## Replication of Paper Results

1. Request access and download the [TUAB](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) and TDBRAIN datasets. Configure the `DATASETS.TUAB.source_root` and the `DATASETS.TDBRAIN.bids_root` directories in your `config.yml` file accordingly. Note that TDBRAIN is provided in BIDS format whereas the TUAB dataset will be converted to BIDS format using a script in the next step.
2. Run through the notebooks in the `scripts` folder in the order indicated by the file names to reproduce the paper results.
