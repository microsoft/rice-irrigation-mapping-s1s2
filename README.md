# Ricemapper: Mapping Rice Irrigation & Sowing from Sentinel Time-Series ([Paper](https://arxiv.org/pdf/2507.08605))

This repository provides a framework for training and inference for the estimation of rice irrigation methods using Sentinel-1 and Sentinel-2 data. The following figure shows the processing pipeline:

![Processing Pipeline](static/processing-pipeline.png)

We use Sentinel-1 timeseries to classify rice field irrigation along two dimensions:
- Sowing
- Irrigation

For Sowing, we classify a plot either into direct seeded rice (DSR) or puddled transplanted rice (PTR).

For Irrigation, we classify a plot either into alternate wetting and drying (AWD) or continuous flooding (CF).

The training data is provided by The Nature Conservancy's Promoting Regenerative and No-burn Agriculture (PRANA) project from North-Western India, namely Punjab. Although this framework can be adapted to other regions, the training data is specific to Punjab for the Kharif season of 2024, and care must be taken to ensure that regions being transferred to share enough similarities in the cropping calendar across the irrigation methods.

We utilize bi-temporal Sentinel-2 data to detect rice field boundaries using the [FTW](https://github.com/fieldsoftheworld/ftw-baselines) pipeline.

## Environment Setup

```bash
mamba create -n rice_mapper python=3.12
conda activate rice_mapper
mamba install conda-forge:gdal
pip install -r requirements.txt
pip install -e.
```

# Recreate Results

This section describes how to recreate the results from the paper. The provided dataset has been de-identified and therefore the coordinates have been removed. This makes it impossible to generate the handcrafted, Presto or Google Satellite Embedding features directly from the original georeferenced polygons. Instead we provide all combinations of features for the 2 best date ranges for each task, as derived in Table 2 of the paper.

## Table 1: Classification Performance

This table attempts to train models on the training dataset for 3-class, Sowing and Irrigation classification.

The available features are:
1. Handcrafted features (HC)
2. Presto features (P)
3. Google Satellite Embeddings (SE)

The available models are:
1. Random Forest (RF)
2. LightGBM (GB)
3. Random Baseline (Random)

The training dataset is available in the form of parquet files containing combinations of features extracted for each plot:
- HC
- HC+P
- HC+P+SE
- HC+SE

These files can be found in `ricemapper/dataset/features/<date_range>/<feature_combination>.parquet`

For DSR and ALL, the best date range (from Table 2) is Jun 1 to Sep 5, 2024 (sampled at f=4days), and for AWD, the best date range is May 1 to Dec 15, 2024 (sampled at f=10days).

### Extract the dataset

Use the following command to untar the dataset:

```
cd <REPO_DIR>/data/
tar -xJf dataset_features.tar.xz
```
This will create a folder called `data/features` with a parquet file for each feature combination.

### Training a model

Set the variables: `OUTPUT_DIR` and `REPO_DIR` in `scripts/train/train.sh`. Run all training scripts together with:
```
cd scripts/train/
chmod +x train.sh
./train.sh --repo-dir [REPOSITORY_PATH] --output-dir [OUTPUT_PATH]
```

The script above trains three models for each task and the performance can vary across models even if the starting seed is the same. Thereofore mileage will vary in terms of performance across models, and you can choose to run more iterations to get a better estimate of the performance.

You can also run custom training jobs — for example to train the best model for DSR:
```
python scripts/train/train_model.py --train_ft_path=<REPO_DIR>/data/features/06-01_09-05_f=4d/train_HC_P.parquet --output_dir=<output_directory> --TASK=ALL_TASKS
```

This will store a model each for 3-class, Sowing and Irrigation classification tasks, in the output directory. By default it will set a `90:10` `train:test` split, with no validation split (what is used in the paper). If you wish to setup your own validation and test splits, you can do so by setting the `SPLIT` and `SPLIT_VAL` parameters in the `train_model.py` script. You can use this to train models for any feature:model combination provided in `Table 1`.

For `Table 2`, one would require the de-identified features for every temporal combination, which is a lot of large files, and therefore not provided. These can be requested from the authors.

## Figures for comparison

To compare the district-wise predictions with the government estimates, you can use the visualization script:

Script: `scripts/visualization/district_results.py`

The CSV file containing the district-wise predictions can be found in `data/Comparison-Govt-Pred.csv`: this contains the estimates from the Govt. of Punjab for 2024, Rice growing area from Han et al. 2022, and the predictions from the models (both masked and non-masked).

### Example Usage:

```bash
# Basic comparison with single model
python scripts/visualization/district_results.py \
    --input data/Comparison-Govt-Pred.csv \
    --output-dir results/district_results \
    --comparison-cols ensemble

# Compare multiple models
python scripts/visualization/district_results.py \
    --input data/Comparison-Govt-Pred.csv \
    --output-dir results/district_results \
    --comparison-cols ensemble ensemble_masked

# Create hybrid column with unmasked districts
python scripts/visualization/district_results.py \
    --input data/Comparison-Govt-Pred.csv \
    --output-dir results/district_results \
    --comparison-cols ensemble_masked_hybrid \
    --unmask-districts "Sri Muktsar Sahib" Fazilka
```

The script generates:
- PNG figures with bar plots and scatter plots comparing government estimates vs. model predictions
- Text files with detailed statistics including correlation coefficients, Jaccard similarity, and Rank Biased Overlap (RBO) scores

For more options, run: `python scripts/visualization/district_results.py --help`

## Error Analysis

To perform detailed error analysis on trained models, use the error analysis script:

Script: `scripts/visualization/error_analysis.py`

This script loads a trained model and evaluates its performance on a test set, generating comprehensive visualizations and metrics including:
- Classification contributions by original class (correct vs incorrect predictions)
- Confusion matrices
- Classification reports (precision, recall, F1-score)
- Feature importance plots (for LightGBM models)

### Example Usage:

```bash
# Basic error analysis for DSR task
python scripts/visualization/error_analysis.py \
    --data-dir /data/panopticon/tnc \
    --output-dir results/error_analysis_DSR \
    --task DSR \
    --lgb

# Error analysis for AWD task
python scripts/visualization/error_analysis.py \
    --data-dir /data/panopticon/tnc \
    --output-dir results/error_analysis_AWD \
    --task AWD \
    --split 0.1 \
    --lgb
```

### Arguments:

- `--data-dir`: Base data directory containing models and features
- `--output-dir`: Directory to save analysis results and plots
- `--task`: Classification task (`DSR` or `AWD`)
- `--split`: Test set split ratio (default: 0.1)
- `--split-val`: Validation set split ratio (default: 0.0)
- `--lgb`: Use LightGBM model (if not set, will use Random Forest)

-------

# Training Workflow

Use this section to train models and run inference on your own data.

## Folder structure

Keep your processed S1 data organized in the following directory structure:

```
<root_directory>/
    s1/
        <orbit>/
            <row>/
                <slice>
                    S1A_*.tif
                    bounds_<row>_<slice>.geojson

    dataset/
        plots/
            <training_plots>.parquet
        s1/
            s1_gamma0/
                <training_plots>.parquet # the S-1 timeseries for each plot is stored here
        features/ # the full feature set for each plot is stored here

        inference/
            districts/
                features/ # the full feature set for each inference plot is stored here

            predictions/ # district-wise predictions are stored here

    models/ # All created models can be stored here

    models_ensemble/ # All created ensemble models can be stored here

    ftw/
        polygons/ # the FTW polygons for each district are stored here
            <district_id>.parquet
```
Where `/data` is the root directory where the data is stored. Create a .env file in the root directory and add the following variables:

```
DATA_DIR=<root_directory>
```

The S1 files should have been preprocessed using the SNAP toolbox from ESA to estimate the gamma0 values for VV and VH bands. Note, sigma0 band values will work, but will likely produce worse edge artifacts across orbit rows.

You also need to extract the bounds of each tile from the metadata of the tif files and store them in a geojson file: `bounds_<row>_<slice>.geojson`.

For detailed instructions on S-1 preprocessing, please refer to the [S-1 preprocessing readme](S1-INSTRUCTIONS.md).

## Generate summary stats

This is the first step in the training workflow: we first summarize the S1 data for each rice growing field as a single time series, by taking the mean of the VV and VH bands for every pixel within the each provided georeferenced polygon.

Script used: `scripts/features/s1_stats.py`

Inputs: \
    - `input_directory` (where the full S1 tiles are stored): `data/s1/gamma0`\
    - `polys_dir` (where the polygons of the rice growing fields are stored): `data/dataset/plots`\
    - `start_date`, \
    - `end_date`, \
    - `frequency`\
Output folder: `dataset/s1/s1_gamma0`\

This script extracts the VV and VH time series for each rice plot provided in the polys_dir, using the S1 data stored in the input_directory, and produces a geojson and parquet file containing all the plots.


## Featurize each plot
This steps generates both handcrafted and Presto features for each plot.

Script used: `scripts/features/featurize_train.py`

Inputs: \
    - `input_path`: `dataset/s1/s1_gamma0/<training_plots>.parquet`\
    - `start_date` (YYYY-MM-DD),  
    - `end_date` (YYYY-MM-DD),\
    - `frequency` (How often to sample the time series, in days)\
    - `output_dir` (where to save the features): `dataset/features/<folder_name>` (e.g. `dataset/features/06-01_08-30_weeks_gamma0_f=7days`)\

Output: `dataset/features/<folder_name>/train_features.parquet` 

## Generating Google Satellite Embedding features

Additionally, use the following script to generate the Google Satellite Embedding features:

Script: `scripts/features/gee_sat_embs.py`

Inputs: \
    - `input-parquet`: Path to input parquet file containing georeferenced polygons\
    - `output-dir`: Directory to save the satellite embeddings\
    - `year`: Year for satellite embeddings (default: 2024)\
    - `head`: Optional number of rows to process from input\

Output: `<output-dir>/satellite_embeddings.parquet`

Example:
```bash
python scripts/features/gee_sat_embs.py \
    --input-parquet dataset/features/06-01_08-30_weeks_gamma0_f=7days/train_features.parquet \
    --output-dir dataset/features/06-01_08-30_weeks_gamma0_f=7days/sat_embs \
    --year 2024
```

## Export S1/ERA5/S2 Time Series with Google Earth Engine

Script: `scripts/features/generate_s1_era5_s2_data_gee.py`

- Requires `.env` entries: `DATA_DIR`, `EE_SERVICE_ACCOUNT`, `EE_KEY` (path to the service-account JSON).
- Loads `<DATA_DIR>/<dir_name>/<geojson_fname>` containing labeled polygons and writes `<DATA_DIR>/dataset/ts_s1_era5_longterm_<date>_<meta>.parquet`.
- Builds Sentinel-1, ERA5, and Sentinel-2 time series per plot through Google Earth Engine with optional orbit filters and retries.

Example:

```bash
python scripts/features/generate_s1_era5_s2_data_gee.py \
    --dir_name november \
    --geojson_fname TNC_plots_fix.geojson \
    --start_date 2024-04-15 \
    --end_date 2024-10-01 \
    --modalities ["s1","era5","s2"] \
    --orbits ["ASCENDING","DESCENDING"] \
    --num_workers 20
```

Tip: pass `--random_subset <N>` to validate settings on a small sample before full runs.

These features can then be concatenated with the other features and used to train a model. The script uses the class `SatelliteEmbeddingExtractor` and function `flatten_sat_embeddings` from `ricemapper.utils.gee.sat_embs` to generate these features.


## Train a model

Script: `scripts/train/train_model.py` 

Inputs: \
    - `train_ft_path`: `dataset/features/<folder_name>/train_features.parquet`\
    - `SPLIT`: proportion of data to use for testing\
    - `SPLIT_VAL`: proportion of data to use for validation\
    - `TASK`: DSR/AWD/ALL/ALL_TASKS (ALL_TASKS: train all tasks serially)\
    - `output_dir`: `models/<folder_name>` (e.g. `models/20240412-80_10_10_weeks`)\
    - `MODELS`: [RF, GB, Random] (RF:Random Forest, GB: LightGBM, Random: Random baseline)

Output:
  A model or models are saved in the output directory.

Example: Use the provided features for training.
```
python scripts/train/train_model.py --SPLIT=0.1 --train_ft_path='train_features_no_coords.parquet' --output_dir='/data/models/20240601_Jun1-Sep15-90_10_f=7d' --TASK=DSR --MODELS=RF
```

### Train an ensemble of models

script: `scripts/train/train_model_ensemble.py`

Inputs: \
    - `train_ft_path`: `dataset/features/<folder_name>/train_features.parquet`\
    - `SPLIT`: proportion of data to use for testing\
    - `SPLIT_VAL`: proportion of data to use for validation\
    - `TASK`: DSR/AWD/ALL\
    - `output_dir`: `models/<folder_name>` (e.g. `models/20240412-80_10_10_weeks`)\
    - `MODELS`: [RF, GB, Random] (RF:Random Forest, GB: LightGBM, Random: Random baseline)
    - `num_models`: number of models to train
    - `CV`: use k-fold cross validation to train the models

Output:
  `num_models` models are saved in the output directory.

Example:
```
python scripts/train/train_model_ensemble.py --SPLIT=0.1  --output_dir='/data/models_ensemble/20250409-90_10' --train_ft_path='<DATA_DIR>/dataset/features/06-01_08-30_weeks_gamma0_f=7days/train_features.parquet' --output_dir='/data/models/20240601_Jun1-Aug30-90_10_f=7d' --TASK=AWD --MODELS=[RF, GB] 
```

# Inference Workflow

The following figure shows the inference workflow:

![Inference Workflow](static/inference-workflow.png)

## Generate summary statistics for each rice growing field

Scripts used: `scripts/features/s1_stats.py`\
Inputs:\
    - `input_directory` (where the S1 data is stored): `data/s1/gamma0`\
    - `polys_dir` (where the polygons of the rice growing fields are stored): `data/ftw/polygons`\
    - `output_dir` (where to save the features): `data/inference/districts/s1_gamma0`\


We first iterate over all the S1 slices/rows and timesteps for each district and save it to a single parquet file for each district. The output folder consists of one geoJson file per district that contains all the rice growing fields in the district.

### Featurization

Scripts used: `scripts/features/featurize.py`\
Inputs:\
    -input_folder: `inference/districts/s1_gamm0/`\
    -output_path: `inference/districts/features/<folder_name>`\
    -start_date: (YYYY-MM-DD)\
    -end_date: (YYYY-MM-DD)\
    -frequency: (How often to sample the time series, in days)\


We iterate over all district files (*.parquet) in the input folder and generate the features for each district.

Example:
```
python scripts/features/featurize.py --frequency=7 --input_path=/data/inference/districts/s1_gamma0 --output_path=data/inference/districts/features/06-01_08-30_f=7days
--start_date=2024-06-01 --end_date=2024-08-30 
```

Important: Pick a date range for the features that matches the date range used for training.

## Inference

Use the following script to run inference on district features:

Script: `scripts/inference/inference_districts.py`

This script runs inference using trained models (single or ensemble) on district features and calculates rice growing areas with optional masking.

### Basic Usage

```bash
python scripts/inference/inference_districts.py \
    --model-dir <path_to_models> \
    --feature-dir <path_to_features> \
    --output-dir <path_to_output>
```

### Full Example with All Options

```bash
python scripts/inference/inference_districts.py \
    --model-dir data/models_ensemble/20250606_Jun1-Sep5_90-10_f=4day/experiment-DSR \
    --feature-dir data/inference/districts/features/06-01_09-05_weeks_gamma0_3_clean \
    --output-dir data/inference/districts/predictions/20250730_Jun1-Sep5_DSR_ensemble \
    --mask-path data/misc/Han2022-paddyRice2021.tif \
    --skip-existing \
    --save-geojson \
    --merge-state
```

### Arguments

- `--model-dir`: Path to directory containing trained model files (.joblib or .txt)
- `--feature-dir`: Path to directory containing district feature parquet files
- `--output-dir`: Path to directory where predictions and results will be saved
- `--mask-path`: (Optional) Path to raster mask file for area calculations
- `--label-cols`: (Optional) Label columns to use for area calculations (default: label_ensemble)
- `--skip-existing`: Skip districts that already have predictions
- `--save-geojson`: Save DSR predictions as separate GeoJSON files per district
- `--merge-state`: Merge all district predictions into a single state-level file
- `--merge-key`: Label column to use when merging state predictions (default: label_ensemble)

### Output

The script generates:
1. Prediction parquet files for each district (`<district>_predictions.parquet`)
2. Area calculation CSV files:
   - `district_areas_detailed.csv`: Detailed area statistics
   - `district_areas_acres.csv`: Summary by district in acres
3. (Optional) GeoJSON files for DSR predictions per district
4. (Optional) State-level merged predictions (`Punjab_predictions.geojson`)

### Masking the rice fields

If you want to mask the rice fields, you can use the following script:

Script: `scripts/ftw/mask_polygons.py`

Inputs:
- `input_path`: `inference/districts/features/<folder_name>`
- `output_path`: `inference/districts/features/<folder_name>_masked`
- `mask_path`: `data/misc/Han2022-paddyRice2021.tif`

The data for Han et al. 2022 is available here: https://zenodo.org/records/5557022

## Generating FTW Polygons

Please [follow the instructions in the FTW repo for inference](https://github.com/fieldsoftheworld/ftw-baselines?tab=readme-ov-file#inference). Save the generated polygons in the `ftw/polygons` folder, with a parquet file per district.

## Data Attribution

This project uses data and features from multiple sources. Below is a comprehensive list of datasets used, their licenses, and attribution requirements:

### Satellite Data

**Sentinel-1 SAR Data**
- **Source:** European Space Agency (ESA) Copernicus Programme
- **License:** Free, full and open access under Copernicus Data Policy
- **Access:** https://scihub.copernicus.eu
- **Citation:** Contains modified Copernicus Sentinel data [Year]
- **Terms:** https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice

**Sentinel-2 Optical Data**
- **Source:** European Space Agency (ESA) Copernicus Programme  
- **License:** Free, full and open access under Copernicus Data Policy
- **Access:** https://scihub.copernicus.eu
- **Citation:** Contains modified Copernicus Sentinel data [Year]
- **Terms:** https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice

### Rice Field Boundaries

**Han et al. 2022 - APRA500 Paddy Rice Dataset**
- **Source:** 500m annual paddy rice maps for monsoon Asia (2000-2021)
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)
- **DOI:** https://doi.org/10.5281/zenodo.5557022
- **Citation:** Han, J. et al. Annual paddy rice planting area and cropping intensity datasets and their dynamics in the Asian monsoon region from 2000 to 2020. Agric. Syst. 200, 103437 (2022).
- **Usage:** Used for masking rice field boundaries in Punjab

### Feature Extraction Models

**Presto (Pretrained Remote Sensing Transformer)**
- **Source:** NASA Harvest, pretrained model for remote sensing time series
- **License:** MIT License
- **Repository:** https://github.com/nasaharvest/presto
- **Citation:** Tseng, G. et al. Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. arXiv:2304.14065 (2023).
- **Usage:** Used for generating learned embeddings from satellite time series

**Google Satellite Embeddings (AlphaEarth Foundations V1)**
- **Source:** Google AlphaEarth Foundations
- **License:** Creative Commons Attribution 4.0 (CC-BY 4.0)
- **Access:** Google Earth Engine Data Catalog
- **Catalog ID:** GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
- **Citation:** Brown, C., Kazmierski, M., Pasquarella, V. et al. AlphaEarth Foundations (in review).
- **Usage:** Used for generating pixel-level embeddings encoding temporal and multi-modal information

### Training Data

**PRANA Project Training Data**
- **Source:** The Nature Conservancy's Promoting Regenerative and No-burn Agriculture (PRANA) Project
- **Region:** Punjab, India (Kharif season 2024)
- **Description:** Field-level data from ~1,400 rice plots including sowing dates, irrigation schedules, and field boundaries
- **License:** NOT PUBLICLY AVAILABLE - Proprietary data collected for this research
- **Note:** The provided dataset in this repository has been de-identified and coordinates removed. Original data cannot be redistributed without permission from The Nature Conservancy.
- **Project Info:** https://www.nature.org/en-us/about-us/where-we-work/india/our-priorities/prana/
- **Citation:** Shah, A. et al. Remote Sensing Reveals Adoption of Sustainable Rice Farming Practices Across Punjab, India. arXiv:2507.08605 (2025).

## Citation

Please cite the following paper if you use this code:
```
@article{shahRemoteSensingReveals2025,
  title = {Remote Sensing Reveals Adoption of Sustainable Rice Farming Practices Across Punjab, India},
  author = {Shah, Ando and Singh, Rajveer and Zaytar, Akram and Tadesse, Girmaw Abebe and Robinson, Caleb and Tafti, Negar and Wood, Stephen A. and Dodhia, Rahul and Ferres, Juan M. Lavista},
  date = {2025-07-11},
  eprint = {2507.08605},
  eprinttype = {arXiv},
  eprintclass = {cs},
  doi = {10.48550/arXiv.2507.08605},
  url = {http://arxiv.org/abs/2507.08605},
}
```

## Trademark Notice
Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.