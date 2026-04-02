# Script Descriptions

In this folder, the scripts for the models are divided into three groups:

## 1. Models with ONLY Images
These include models trained using only image data at different spatial scales (0.5 km, 1 km, 3 km, and 5 km).

- Single-scale models:
  - `MResNet18_extcv_single_config_ONLY_images.py`
  - Refit using best hyperparameters: `MResNet18_refit_images_ONLY.py`

- Multi-scale model (all radii combined):
  - Training: `MResNet18_multiRadii_extcv_ONLY_images.py`
  - Refit: `MResNet18_multiRadii_ONLY_images_refit.py`

## 2. Models with ONLY Tabular Data
This group includes models trained using only tabular features:

- `GLM.py`, `GLM_reg.py`, `TabularDNN.py`, `XGB.py`
- Refit script for the DNN: `TabularDNN_refit.py`

## 3. Models with Tabular + Images
These correspond to experiments combining tabular features with a single image scale:

- Training: `MResNet18_extcv_single_config.py`
- Refit: `MResNet18_refit_single_config.py`

---

## Additional Experiments

The folder `Study_by_zones_extensive_CV` contains examples of different feature sets evaluated in the initial experiment. This setup was later extended to:

- `Study_by_zones_extensive_CV_1` to `Study_by_zones_extensive_CV_4`

These include `.sh` scripts for running experiments under different configurations:

### Only Images
- Script for generating `.sh` files: `slurm_sh_files_creation.py`
- Example refit script for one outer fold: `refit0.sh` (extended to `outer1`–`outer5`)
- Configurations include:
  - All radii: `ALL_radii`
  - Single scales: `Neigh_0.5`, `Neigh_1`, `Neigh_3`, `Neigh_5`

### With Latitude and Longitude
Includes `.sh` files that incorporate spatial coordinates (lat, long), as well as models using learned embeddings:

- ResNet18 embeddings:
  - `GLM_reg_ResNet18Emb`, `XGB_ResNet18Emb`
- Nomic v1.5 embeddings:
  - `GLM_reg_nomic_v15Emb`, `XGB_nomic_v15Emb`

### With Latitude, Longitude, and OSM-Corine Features
Folder: `With_lat_long_osm14corine2000`

- Includes models with OSM14 + Corine 2000 features at 0.5 km
- `Radious_0.5km`: base tabular (no-image) models
- Based on preliminary experiments, the best image scale was 3 km:
  - `Image_DNN/Neigh_3`: ResNet18 models combined with environmental features

Other radii folders (`Radious_1km`, `Radious_3km`, `Radious_5km`, `Radious_ALL`) follow the same structure.

---

## Additional Configurations

Analogous to the previous subloders of `Study_by_zones_extensive_CV`, we created additional experiment groups:

- `With_lat_long_postcode2`
- `With_lat_long_postcode2_osm14_corine2000`
- `With_osm14corine2000`
- `With_postcode2`
- `With_postcode2_osm14corine2000`
- `Without_location`
