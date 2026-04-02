# Script Descriptions

In this folder, the scripts for the models are devided in 3 groupps, namely:
1. Models with ONLY images: These includes the ones using for training only one scale of images, 0.5km, 1km, 3km or 5km, *MResNet18_extcv_single_config_ONLY_images.py* and the correctponding script for refitting using the best hyperparameters found *MResNet18_refit_images_ONLY.py*. And the model that uses all the radious scales at the same time, for training, *MResNet18_multiRadii_extcv_ONLY_images.py* and for refitting *MResNet18_multiRadii_ONLY_images_refit.py*.
2. Models with Only tabular data: This contains *GLM.py*, *GLM_reg.py*, *TabularDNN.py*, *XGB.py* and the script used for refitting the DNN model using the best set of hyperparameters, *TabularDNN_refit.py*.
3. Models with Tabular and Images: Given experiments on the model using one of the images scale and another tabular features, taraining and refiting stage given in *MResNet18_extcv_single_config.py* and *MResNet18_refit_single_config.py*

Aditionally we include in the folder **Study_by_zones_extensive_CV** some examples of the different sets of features that were evaluated only for the first experiment, similarly was extended to have **Study_by_zones_extensive_CV_1** to **Study_by_zones_extensive_CV_4**. Specifically we include the .sh files creation for the case of:
- **Only images**: Including the .sh *slurm_sh_files_creation.py* file creation and the refit sh script for one outerfold (outer0) given the best hyperparameters found, *refit0.sh*, the latter we extended for all the outer, this is outer1 to outer5, for the case of contempling the all the scale of images (**ALL_radii**) or only the scale of 0.5km (**Neigh_0.5**). We had this .sh files for every option of scale remined, this is ***Neigh_1**, **Neigh_3**.
- **With_lat_long**: includes the .sh files contempleting the inclusion of the coordinates lat, long, this includes also the models contemplating the inclusion of embeddings given by the modified ResNet18 or Nomicv15, i.e., **GLM_reg_ResNet18Emb**, **XGB_ResNet18Emb**, and, **GLM_reg_nomic_v15Emb**, **XGB_nomic_v15Emb**, respectively.
- **With_lat_long_osm14corine2000** this folder includes the codes for the models including the osm14corine 2000 features at 0.5km.
   In the subfolder **Radious_0.5km**, the no-image base models are included. And according with preliminary experiments for this experiment the best scale for the images was at 3km, thus the folder **Image_DNN/Neigh_3** includes the modified resnet18 script for training and refitting using also the environmental features. The subfolders **Radious_1km**, **Radious_3km**, **Radious_5km**, and **Radious_ALL** are constructed similarlly as **Radious_0.5km**

Analogous to the previous subloders to **Study_by_zones_extensive_CV**, we constructed per each of the subgroups: With_lat_long_posc2, With_lat_long_postc2_osm14_corine2000, With_osm14_corine2000, With_postcode2, With_postcode2_osm14_corine2000, and, Without_location.

# Script Descriptions

In this folder, scripts are divided into three groups:

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

The folder `Study_by_zones_extensive_CV` contains examples of different feature sets evaluated in the initial experiments. This setup was later extended to:

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

### With Latitude, Longitude, and OSM/Corine Features
Folder: `With_lat_long_osm14corine2000`

- Includes models with OSM14 + Corine 2000 features at 0.5 km
- `Radious_0.5km`: base tabular (no-image) models
- Based on preliminary experiments, the best image scale was 3 km:
  - `Image_DNN/Neigh_3`: ResNet18 models combined with environmental features

Other radii folders (`Radious_1km`, `Radious_3km`, `Radious_5km`, `Radious_ALL`) follow the same structure.

---

## Additional Configurations

Analogous to `Study_by_zones_extensive_CV`, we created additional experiment groups:

- `With_lat_long_postcode2`
- `With_lat_long_postcode2_osm14_corine2000`
- `With_osm14corine2000`
- `With_postcode2`
- `With_postcode2_osm14corine2000`
- `Without_location`
