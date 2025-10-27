---
language:
- en
license: mit
library_name: scikit-learn
tags:
- remote-sensing
- agriculture
- rice-classification
- sentinel-1
- sentinel-2
- time-series
datasets:
- sentinel-1
- sentinel-2
- prana-project
metrics:
- accuracy
- f1
- precision
- recall
---

# Model Card for Ricemapper

Ricemapper is a framework for classifying rice field irrigation methods and sowing practices using Sentinel-1 and Sentinel-2 satellite time-series data. The model predicts both sowing methods (Direct Seeded Rice vs Puddled Transplanted Rice) and irrigation practices (Alternate Wetting and Drying vs Continuous Flooding) for rice fields.

## Model Details

### Model Description

Ricemapper uses machine learning classifiers (Random Forest and LightGBM) trained on multi-source features extracted from Sentinel-1 SAR time-series data. The framework combines handcrafted spectral-temporal features, Presto pretrained embeddings, and Google Satellite Embeddings to classify rice cultivation practices across two dimensions: sowing methods and irrigation regimes.

- **Developed by:** Ando Shah, Rajveer Singh, Akram Zaytar, Girmaw Abebe Tadesse, Caleb Robinson, Negar Tafti, Stephen A. Wood, Rahul Dodhia, Juan M. Lavista Ferres
- **Model type:** Classification (Random Forest, LightGBM)
- **Language(s) (NLP):** N/A (Remote Sensing Model)
- **License:** MIT
- **Finetuned from model:** Presto (Pretrained Remote Sensing Transformer) for feature extraction

### Model Sources

- **Repository:** https://github.com/microsoft/rice-irrigation-mapping-s1s2
- **Paper:** https://arxiv.org/pdf/2507.08605
- **Demo:** N/A

## Uses

### Direct Use

The model can be directly used to classify rice fields in Punjab, India (and potentially other regions with similar growing conditions) into:
- **Sowing classes:** Direct Seeded Rice (DSR) or Puddled Transplanted Rice (PTR)
- **Irrigation classes:** Alternate Wetting and Drying (AWD) or Continuous Flooding (CF)
- **Combined 3-class classification** capturing both dimensions

Primary use cases include monitoring sustainable agricultural practices, tracking AWD adoption for water conservation, and supporting agricultural policy decisions.

### Downstream Use

The model outputs can be integrated into larger agricultural monitoring systems for:
- Regional water resource management
- Carbon footprint estimation for rice cultivation
- Agricultural subsidy and incentive program verification
- Climate adaptation strategy development
- Precision agriculture advisory systems

### Out-of-Scope Use

The model is not suitable for:
- Rice cultivation outside the Kharif (monsoon) season timeframe
- Regions with different climate, soil, or cropping calendars than Punjab
- Other crop types beyond rice
- Individual plant-level classification
- Real-time monitoring (requires complete seasonal time-series)
- Areas without adequate Sentinel-1 coverage

## Bias, Risks, and Limitations

**Temporal Limitations:** The model is trained specifically on Kharif season 2024 data (June-December). Performance may degrade for other seasons or years with different weather patterns.

**Geographic Limitations:** Training data comes exclusively from Punjab, India. Transfer to other rice-growing regions requires careful validation due to differences in cropping calendars, climate, and agricultural practices.

**Data Requirements:** Requires complete Sentinel-1 time-series at 4-10 day intervals. Missing data due to orbit gaps or processing issues may impact predictions.

**Class Imbalance:** Performance may vary across classes depending on their representation in the training data.

**Boundary Dependencies:** Accuracy depends on the quality of field boundary delineation from FTW (Fields of the World) segmentation.

### Recommendations

Users should:
- Validate model performance on local ground-truth data before operational deployment
- Consider the specific date ranges used during training (June 1 - September 5 for DSR/ALL tasks; May 1 - December 15 for AWD tasks)
- Be aware that predictions represent field-level aggregates, not pixel-level classifications
- Use ensemble models for more robust predictions
- Apply rice field masks (e.g., Han et al. 2022) to reduce false positives in non-rice areas
- Collect regional ground truth data for fine-tuning when transferring to new regions

## How to Get Started with the Model

```python
import joblib
import pandas as pd
from ricemapper.features import extract_features

# Load trained model
model = joblib.load('models/ricemapper_dsr_model.joblib')

# Load and prepare your features
features_df = pd.read_parquet('your_features.parquet')

# Make predictions
predictions = model.predict(features_df)

# Get probabilities
probabilities = model.predict_proba(features_df)
```

For complete workflow including feature extraction:
```bash
# Extract S1 statistics
python scripts/features/s1_stats.py --input_directory data/s1/gamma0 \
    --polys_dir data/plots --output_dir data/dataset/s1

# Generate features
python scripts/features/featurize_train.py \
    --input_path data/dataset/s1/plots.parquet \
    --start_date 2024-06-01 --end_date 2024-09-05 \
    --frequency 4 --output_dir data/features

# Train model
python scripts/train/train_model.py \
    --train_ft_path data/features/train_features.parquet \
    --output_dir models/ --TASK DSR --MODELS RF
```

## Training Details

### Training Data

Training data consists of approximately 1,400 rice plots from Punjab, India collected during the Kharif season 2024 through The Nature Conservancy's PRANA project. The dataset includes:
- Field boundaries (georeferenced polygons)
- Sowing dates and methods
- Irrigation schedules and practices
- Complete Sentinel-1 VV and VH time-series

**Note:** The original training data is proprietary and not publicly available. A de-identified version with coordinates removed is provided in this repository.

**External datasets used:**
- Sentinel-1 SAR imagery (ESA Copernicus)
- Sentinel-2 optical imagery (ESA Copernicus)
- Han et al. 2022 APRA500 paddy rice maps for field masking
- Presto pretrained embeddings
- Google AlphaEarth Satellite Embeddings

### Training Procedure

#### Preprocessing

1. **S1 Preprocessing:** Sentinel-1 data preprocessed using ESA SNAP toolbox to generate gamma0 values for VV and VH bands
2. **Time-series Extraction:** Mean VV and VH values computed for each field polygon across the time series
3. **Feature Engineering:** Three feature types extracted:
   - Handcrafted features (HC): Statistical and spectral-temporal features from VV/VH time-series
   - Presto embeddings (P): Learned features from pretrained transformer
   - Google Satellite Embeddings (SE): Pixel-level multi-temporal embeddings
4. **Temporal Sampling:** Time-series sampled at 4-day intervals (DSR/ALL) or 10-day intervals (AWD)
5. **Data Splits:** 90% training, 10% testing (no validation set used in paper results)

#### Training Hyperparameters

- **Training regime:** fp32
- **Models:** Random Forest and LightGBM with default parameters
- **Feature combinations tested:** HC, HC+P, HC+P+SE, HC+SE
- **Date ranges optimized:** Multiple temporal windows tested per task
- **Ensemble:** Multiple models trained with different random seeds for ensemble predictions

#### Speeds, Sizes, Times

Training times vary by model complexity:
- Random Forest: Minutes to hours depending on feature set
- LightGBM: Similar to Random Forest
- Full pipeline (feature extraction + training): ~1-2 hours per experiment

Model sizes: 10-500 MB depending on feature set and model type

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Held-out 10% test set from the PRANA training dataset, maintaining geographic and temporal distribution of the full dataset.

#### Factors

Evaluation disaggregated by:
- Classification task (DSR vs AWD vs 3-class)
- Feature combination (HC, HC+P, HC+SE, HC+P+SE)
- Model type (Random Forest vs LightGBM)
- Temporal window (various date ranges tested)

#### Metrics

- **Accuracy:** Overall classification accuracy
- **Precision, Recall, F1-score:** Per-class performance metrics
- **Confusion matrices:** Detailed error analysis
- **Feature importance:** For LightGBM models
- **District-level validation:** Comparison with government agricultural statistics
- **Jaccard similarity and Rank Biased Overlap (RBO):** For district-level predictions

### Results

Best performing configurations (from Table 2 in paper):
- **DSR task:** June 1 - September 5, f=4 days, HC+P features
- **AWD task:** May 1 - December 15, f=10 days, HC+P features
- **ALL tasks:** June 1 - September 5, f=4 days, HC+P features

Detailed performance metrics available in paper Table 1.

#### Summary

The framework successfully classifies rice cultivation practices in Punjab with high accuracy using multi-source satellite features. Ensemble models combining Random Forest and LightGBM improve robustness. The addition of learned embeddings (Presto, Google Satellite) to handcrafted features generally improves performance. District-level predictions show strong correlation with government agricultural statistics.

## Model Examination

Feature importance analysis (for LightGBM) reveals:
- Early season temporal features most important for DSR classification
- Mid-to-late season features critical for AWD detection
- VH band features generally more informative than VV
- Presto embeddings capture temporal patterns not easily extracted through handcrafted features

Classification contribution analysis shows model performance across original class labels, identifying specific confusion patterns between classes.

## Environmental Impact

Carbon emissions minimal for this project as training performed on standard CPU infrastructure.

## Technical Specifications

### Model Architecture and Objective

**Architecture:** Ensemble of Random Forest and LightGBM classifiers

**Objective:** Multi-class classification for sustainable rice agriculture monitoring
- Binary classification for sowing method (DSR vs PTR)
- Binary classification for irrigation practice (AWD vs CF)
- 3-class joint classification

**Input:** Feature vectors of length 100-1000+ (depending on feature combination) derived from Sentinel-1 time-series

**Output:** Class predictions and probability scores

### Compute Infrastructure

#### Hardware

Standard workstations with:
- CPU: Multi-core processors (8+ cores)
- RAM: 32+ GB
- Storage: SSD for fast data access

No GPU required for training or inference.

#### Software

- Python 3.12
- scikit-learn (Random Forest)
- LightGBM
- GDAL for geospatial processing
- pandas, numpy for data manipulation
- Presto for embedding extraction
- Google AlphaEarth Satellite Embeddings

## Citation

**BibTeX:**

```bibtex
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

**APA:**

Shah, A., Singh, R., Zaytar, A., Tadesse, G. A., Robinson, C., Tafti, N., Wood, S. A., Dodhia, R., & Lavista Ferres, J. M. (2025). Remote Sensing Reveals Adoption of Sustainable Rice Farming Practices Across Punjab, India. arXiv preprint arXiv:2507.08605.

## Glossary

- **DSR (Direct Seeded Rice):** Rice cultivation method where seeds are sown directly in the field rather than transplanting seedlings
- **PTR (Puddled Transplanted Rice):** Traditional method involving transplanting seedlings into puddled (flooded and tilled) fields
- **AWD (Alternate Wetting and Drying):** Water-saving irrigation technique involving periodic drying of fields
- **CF (Continuous Flooding):** Traditional irrigation maintaining continuous standing water in fields
- **Kharif season:** Monsoon cropping season in India (typically June-November)
- **Sentinel-1:** ESA satellite providing C-band SAR imagery
- **Sentinel-2:** ESA satellite providing optical multispectral imagery
- **Presto:** Pretrained transformer model for remote sensing time-series
- **FTW:** Fields of the World - deep learning pipeline for field boundary detection
- **Gamma0:** Radiometrically terrain-corrected SAR backscatter coefficient

## More Information

For detailed information on:
- S1 preprocessing: See S1-INSTRUCTIONS.md in the repository
- Feature extraction methodology: See scripts/features/ directory
- Training procedures: See scripts/train/ directory
- Inference workflows: See scripts/inference/ directory

For questions about the PRANA project or data access, visit: https://www.nature.org/en-us/about-us/where-we-work/india/our-priorities/prana/

## Model Card Authors

Akram Zaytar

## Model Card Contact

For questions or issues, please open an issue on the GitHub repository or contact the corresponding author through the paper.