# Third-Party Licenses

This project includes several third-party open-source libraries, each with its own license. This file lists the licenses of the dependencies we use.

## Python Dependencies

### Development and Utilities
- **jupyter**: BSD 3-Clause License
  - https://github.com/jupyter/jupyter/blob/master/LICENSE
- **ipykernel**: BSD 3-Clause License
  - https://github.com/ipython/ipykernel/blob/main/LICENSE
- **python-dotenv**: BSD 3-Clause License
  - https://github.com/theskumar/python-dotenv/blob/main/LICENSE
- **fire**: Apache License 2.0
  - https://github.com/google/python-fire/blob/master/LICENSE

### Scientific Computing and Data Processing
- **scikit-learn**: BSD 3-Clause License
  - https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
- **scipy**: BSD 3-Clause License
  - https://github.com/scipy/scipy/blob/main/LICENSE.txt
- **numpy**: BSD 3-Clause License
  - https://github.com/numpy/numpy/blob/main/LICENSE.txt
- **pandas**: BSD 3-Clause License
  - https://github.com/pandas-dev/pandas/blob/main/LICENSE
- **openpyxl**: MIT License
  - https://github.com/theorchard/openpyxl/blob/master/LICENCE.rst
- **pyarrow**: Apache License 2.0
  - https://github.com/apache/arrow/blob/main/LICENSE.txt

### Geospatial and Imaging
- **rasterio**: BSD 3-Clause License
  - https://github.com/rasterio/rasterio/blob/master/LICENSE.txt
- **geopandas**: BSD 3-Clause License
  - https://github.com/geopandas/geopandas/blob/main/LICENSE.txt
- **shapely**: BSD 3-Clause License
  - https://github.com/shapely/shapely/blob/main/LICENSE.txt
- **rioxarray**: Apache License 2.0
  - https://github.com/corteva/rioxarray/blob/master/LICENSE
- **exactextract**: Apache License 2.0
  - https://github.com/isciences/exactextract/blob/master/LICENSE
- **earthengine-api**: Apache License 2.0
  - https://github.com/google/earthengine-api/blob/master/LICENSE

### Visualization
- **matplotlib**: Matplotlib License (BSD-compatible)
  - https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE
- **seaborn**: BSD 3-Clause License
  - https://github.com/mwaskom/seaborn/blob/master/LICENSE

### Deep Learning and Machine Learning
- **torch (PyTorch)**: BSD-style license
  - https://github.com/pytorch/pytorch/blob/master/LICENSE
- **einops**: MIT License
  - https://github.com/arogozhnikov/einops/blob/master/LICENSE
- **lightgbm**: MIT License
  - https://github.com/microsoft/LightGBM/blob/master/LICENSE
- **xgboost**: Apache License 2.0
  - https://github.com/dmlc/xgboost/blob/master/LICENSE
- **optuna**: MIT License
  - https://github.com/optuna/optuna/blob/master/LICENSE
- **tslearn**: BSD 2-Clause License
  - https://github.com/tslearn-team/tslearn/blob/main/LICENSE

### Code Snippets and Adaptations
- **Sentinel-2 cloud masking functions**: Apache License 2.0
  - License: https://www.apache.org/licenses/LICENSE-2.0
  - Source: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
  - Used in: `./ricemapper/utils/gee/s2.py`
  - Note: Code adapted from Google Earth Engine community tutorial