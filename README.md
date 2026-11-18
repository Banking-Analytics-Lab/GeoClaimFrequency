# GeoClaimFrequency 🛰️🗺️📊

Code for the paper:

> Sherly Alfonso-Sánchez, Cristián Bravo, Kristina G. Stankova (2026). Revealing Geography-Driven Signals in Zone-Level Claim Frequency Models: An Empirical Study using Environmental and Visual Predictors. *arXiv preprint arXiv:2604.21893*. <https://doi.org/10.48550/arXiv.2604.21893>

This repository contains the code of the research project conducted in 2025-2026: *"Revealing Geography-Driven Signals in Zone-Level Claim Frequency Models: An Empirical Study using Environmental and Visual Predictors"*; this work was done by Ph.D. candidate Ms. Sherly Alfonso-Sánchez, under the supervision of Dr. Kristina G. Stankova and Dr. Cristián Bravo Roman (2026). The code aggregates the beMTPL97 portfolio by postcode, builds environmental features from OpenStreetMap 2014 and CORINE Land Cover 2000 and image embeddings from 1995 orthoimagery, and fits Poisson GLMs, regularized GLMs, XGBoost, tabular neural networks and ResNet18 models, tested on six stratified outer folds of postcodes.

### Abstract

Geographic context is often considered relevant for assessing motor insurance risk, yet public actuarial datasets typically provide limited location identifiers, constraining how this information can be incorporated in claim-frequency models. This study examines how geographic information from alternative data sources can be incorporated into actuarial models for Motor Third Party Liability (MTPL) claim prediction under such constraints. Using the BeMTPL97 dataset [(CAS Dataset Manual)](https://cas.uqam.ca/pub/web/CASdatasets-manual.pdf), we adopt a zone-level modeling framework and evaluate predictive performance on postcodes not observed during training. Geographic information is introduced through two channels: environmental indicators from OpenStreetMap (2014) and CORINE Land Cover 2000, and orthoimagery from 1995 released by the Belgian National Geographic Institute for academic use. We evaluate the predictive contribution of coordinates, environmental features, and image embeddings across three baseline models: generalized linear models (GLMs), regularized GLMs, and gradient-boosted trees, while raw imagery is modeled using convolutional neural networks. Our results show that augmenting actuarial variables with constructed geographic information improves predictive accuracy. Across experiments, both linear and tree-based models benefit most from combining coordinates with environmental features extracted at 5 km scale, while smaller neighborhoods also improve baseline specifications. Generally, image embeddings do not improve performance when structured environmental features are available; however, when such features are absent, pretrained vision-transformer embeddings enhance accuracy and stability for regularized GLMs. Our results show the predictive value of geographic information in zone-level MTPL frequency models depends less on model complexity than on how geography is represented, and illustrate that geographic context can be incorporated despite limited individual-level spatial information.

### Repository structure

The scripts are organized in the following order. Each folder has a `Readme.md` that describes its scripts.

1. **Preprocessing and Aggregation** (`Preprocessing and Agreggation/`): filtering of beMTPL97 and aggregation by postcode.
2. **Alternative Geographic Data** (`Alternative Geo Data/`): OpenStreetMap 2014 and CORINE 2000 features per postcode.
3. **Data Splits by Experiment** (`Data Splits by Experiment/`): merge of the features, six stratified folds, ResNet18 and Nomic v1.5 image embeddings.
4. **Zone-Level Frequency Models** (`Zone Level Frequency Models/`): the model scripts, and the Slurm scripts of one experiment in `Study_by_zones_extensive_CV/`.

The repository holds one of the five repeated experiments of the paper (folds drawn with seed 611 in `Division_6.py`). The folders of the other four experiments (`Data_creation_1` to `Data_creation_4`, `Study_by_zones_extensive_CV_1` to `Study_by_zones_extensive_CV_4`) and several feature-set folders are not included; see `Data Splits by Experiment/Readme.md` and `Zone Level Frequency Models/Readme.md`.

## Requirements

The Python version and the package versions used for the paper were not recorded. `requirements.txt` lists the packages that the code imports, without versions.

```bash
pip install -r requirements.txt
```

The experiments ran as Slurm jobs on clusters of the Digital Research Alliance of Canada. The Slurm scripts load `module load gcc arrow/19.0.1` (`Data_merge_all.sh` loads `python/3.11 proj gdal geos`) and activate a virtual environment in the home folder (`~/p3_env_nvl_test` or `~/p3_env_gnn`). Replace these lines with your own environment. Where a script had an allocation, it now reads `#SBATCH --account=<your-allocation>`; replace it with yours.

The ResNet18, embedding and tabular neural-network jobs request `--gpus=a100_1g.5gb:1` and 8 GB of memory. The other jobs request CPUs only; the GLM, regularized GLM and XGBoost jobs request 1 CPU and 8 GB. The `--time` values in the scripts are requested limits; the run times were not recorded.

### Paths

The scripts read their paths from these environment variables. The defaults are relative to the folder where you run or submit a script.

| Variable | Default | Content |
|---|---|---|
| `INPUT_DIR` | `data` | Input files, see `data/README.md` |
| `IMAGES_DIR` | `data/Images_Ortho_95` | Orthoimage tiles |
| `WORK_DIR` | `data/Data_creation` | Datasets built by the pipeline: merged features, folds, embeddings |
| `RESULTS_DIR` | `results` | Model outputs and the Slurm scripts written by the generator scripts |
| `CODE_DIR` | `.` | Root of this repository, used to call the Python scripts |
| `BEMTPL_FILTERED_CSV` | `beMTPL97_filtered.csv` | Input of `creation_data_aggregated.py` |

Submit the Slurm scripts from the repository root. Two scripts call their Python file by its bare name and are submitted from their own folder: `make_splits.sh` and `Creation_nomic3km_emb.sh`. For these, set the variables to absolute paths. `#SBATCH --output` lines cannot read variables, so they hold paths under `results/`, relative to the submission folder. Slurm does not create that folder; create it before you submit. The generator scripts write the resolved paths into the Slurm scripts they produce.

## Data

The code uses the public beMTPL97 portfolio from the R package CASdatasets, an OpenStreetMap extract of Belgium from 2014 (Geofabrik), CORINE Land Cover 2000 (Copernicus Land Monitoring Service) and the 1995 orthoimagery of the Belgian National Geographic Institute, obtained from Cartesius.be for academic use. The imagery is not redistributed here. Two derived tables are in the repository: `Preprocessing and Agreggation/beMTPL97_filtered.csv` and `Preprocessing and Agreggation/BeMTPL_Agg.csv`.

`data/README.md` lists every input file, its source and terms, and where the code expects it.

## Training

Run the steps in this order. Commands run from the repository root unless a step says otherwise.

1. Filter the portfolio. Open `Preprocessing and Agreggation/Preprocessing_Belgian.ipynb` in its folder. It reads `../data/beMTPL97.csv`, keeps the policies with at most three claims, and writes `beMTPL97_filtered.csv` next to the notebook.

2. Aggregate by postcode. The script writes `BeMTPL_Agg.csv` in the current folder; step 5 reads it from `$INPUT_DIR/`.

   ```bash
   cd "Preprocessing and Agreggation"
   python creation_data_aggregated.py
   ```

3. Extract the unique locations. `Preprocessing and Agreggation/Extract_geo_location_options.ipynb` reads `beMTPL97_filtered.csv` next to it and writes `unique_locations_BeMTPL97.csv`. The later scripts read this file from `$INPUT_DIR/` and `$INPUT_DIR/OSM_2014_CORINE_2000/`.

4. Build the environmental features at 0.5, 1, 3 and 5 km. The script writes `osm_features_OSM2014_maskCLC2000_faster3.csv` in the current folder; step 5 reads it from `$INPUT_DIR/OSM_2014_CORINE_2000/`.

   ```bash
   python "Alternative Geo Data/osm14corine2000_feature_extract.py"
   ```

5. Merge the features with the aggregated data. Writes `$WORK_DIR/With_osm14_corine2000/all/Data_all_osm14_clc2000_ALL.csv`.

   ```bash
   sbatch "Data Splits by Experiment/With_osm14_corine2000/Data_merge_all.sh"
   ```

6. Draw six stratified folds (seed 611). `Division_6.py` writes `alldata_with_fold_id.csv`, `D1.csv` to `D6.csv`, a fold summary and plots in the current folder. The later scripts read `$WORK_DIR/Stratified_division/alldata_with_fold_id.csv`.

   ```bash
   cd "Data Splits by Experiment/Data_creation/Stratified_division"
   sbatch make_splits.sh
   ```

7. Build the image embeddings from the 3 km tiles.

   Nomic v1.5: in `Data Splits by Experiment/Data_creation/Embeddings_nomic_v15/`, run `Saving_nominic_v15_weights.py` on a node with internet access. It saves `nomic-ai/nomic-embed-vision-v1.5` to `./nomic_v15_local`. `Creation_nomic3km_emb.sh` expects that folder at `$WORK_DIR/Embeddings_nomic_v15/nomic_v15_local` and writes `augmented_with_nomic3km.parquet` and `.csv` there.

   ```bash
   cd "Data Splits by Experiment/Data_creation/Embeddings_nomic_v15"
   python Saving_nominic_v15_weights.py
   sbatch Creation_nomic3km_emb.sh
   ```

   ResNet18, one script per outer fold (folds 0 and 1 are in the repository, folds 2 to 5 are not). Each needs the refitted images-only ResNet18 at 3 km for its fold, `$RESULTS_DIR/ONLY_images/Neigh_3/Refit_model/outer<k>/refit_model_outer<k>.pth` and `image_norm_outer<k>.json`. The refit script for 3 km images only is not in the repository; `ONLY_images/Neigh_0.5/Refit_model/outer0/refit0.sh` shows the 0.5 km version.

   ```bash
   sbatch "Data Splits by Experiment/Data_creation/Augmented_data_with_embeddings/outer0/Data_aug_creation0.sh"
   sbatch "Data Splits by Experiment/Data_creation/Augmented_data_with_embeddings/outer1/Data_aug_creation1.sh"
   ```

   These write `$RESULTS_DIR/Embeddings_ResNet18/Augmented_data_with_embeddings/outer<k>/data_withfolds_id_withEmb_outer<k>.csv`. The GLM and XGBoost scripts that use the embeddings read `$WORK_DIR/Embeddings_Resnet18/outer<k>/data_withfolds_id_withEmb_outer<k>.csv`. Copy the files from the first place to the second.

8. Fit the GLM, regularized GLM and XGBoost models. Each script is a Slurm array over the six outer folds.

   ```bash
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/GLM/GLM.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/GLM_reg/GLM_reg.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/XGB/XGB.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/GLM_reg_ResNet18Emb/GLM_reg_with_ResNet18_emb.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/XGB_ResNet18Emb/XGB_with_ResNet18_emb.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/GLM_reg_nomic_v15Emb/NomicEmb_GLM_reg.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/XGB_nomic_v15Emb/NomicEmb_XGB.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Radious_0.5km/GLM/GLM.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Radious_0.5km/GLM_reg/GLM_reg.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Radious_0.5km/XGB/XGB.sh"
   ```

9. Fit the neural networks. Each generator writes one Slurm script per outer fold and epoch budget (15, 30, 60) under `$RESULTS_DIR`. Each generated script is a Slurm array over rows 0 to 7 of the grid `$INPUT_DIR/hyperparam_grid_ep<epochs>.csv`, which is not in the repository (see `data/README.md`). Submit the generated scripts with `sbatch`.

   ```bash
   python "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/DNN/slurm_make.py"
   python "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Radious_0.5km/DNN/slurm_make.py"
   python "Zone Level Frequency Models/Study_by_zones_extensive_CV/ONLY_images/Neigh_0.5/slurm_files_creation.py"
   python "Zone Level Frequency Models/Study_by_zones_extensive_CV/ONLY_images/ALL_radii/slurm_sh_files_creation.py"
   python "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/Image_DNN/Neigh_3/slurm_files_creation.py"
   python "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Image_DNN/Neigh_3/slurm_files_creation.py"
   ```

   Then refit each model with its selected configuration on the outer fold. The examples are for outer fold 0, with the hyperparameters written in each script. How these were chosen from the inner cross-validation summaries is not documented. `TabularDNN_refit.py` has no Slurm script in the repository.

   ```bash
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/ONLY_images/Neigh_0.5/Refit_model/outer0/refit0.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/ONLY_images/ALL_radii/Refit_model/outer0/refit0.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long/Image_DNN/Neigh_3/Refit_model/outer0/refit_outer0.sh"
   sbatch "Zone Level Frequency Models/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Image_DNN/Neigh_3/Refit_model/outer0/refit_outer0.sh"
   ```

## Evaluation

Each model script tests on the outer fold given by `--outer_fold` and writes its files into `--out_dir`:

- `GLM.py`: `cv_metrics_outer<k>.csv` and `outer_test_metrics_outer<k>.csv`.
- `GLM_reg.py` and `XGB.py`: `inner_summary_outer<k>.csv`, `outer_eval_outer<k>.csv` and `outer_predictions_outer<k>.csv`; `XGB.py` also writes `inner_details_outer<k>.csv`. `GLM_reg.py` prints the non-zero coefficients of the refitted model to the job log.
- Neural-network scripts for the inner cross-validation (`TabularDNN.py`, `MResNet18_extcv_single_config.py`, `MResNet18_extcv_single_config_ONLY_images.py`, `MResNet18_multiRadii_extcv_ONLY_images.py`): `inner_details_outer<k>.csv` and `inner_summary_outer<k>.csv`.
- Refit scripts: `outer_eval_outer<k>.csv`, `outer_predictions_outer<k>.csv`, `refit_model_outer<k>.pth` and a learning-curve PNG; the image models also write `image_norm_outer<k>.json`.

How the per-fold files are combined into the tables of the paper is not documented: the repository has no aggregation script.

## Pretrained models

Not released. The refit scripts write the fitted weights (`refit_model_outer<k>.pth`), but none are in the repository. The image models start from `resnet18_weights.pth`, a state dictionary for torchvision's `resnet18` that is not in the repository; its source is not documented. The Nomic v1.5 model is downloaded from Hugging Face by `Saving_nominic_v15_weights.py`.

## Results

The paper reports test RMSE per outer fold, with its mean and standard deviation. The table maps each results table to the scripts in this repository that produce its per-fold runs. No numbers are reproduced here.

| Table or figure in the paper | Command | Expected result |
|---|---|---|
| Table 5 (GLM by predictor set) | `GLM.sh` in `With_lat_long/GLM/` and `With_lat_long_osm14corine2000/Radious_0.5km/GLM/`; other predictor sets not in the repository; aggregation not documented | See the paper |
| Table 6 (regularized GLM) | `GLM_reg.sh` in the same two folders; aggregation not documented | See the paper |
| Table 7 (XGBoost) | `XGB.sh` in the same two folders; aggregation not documented | See the paper |
| Table 8 (ResNet18, images only or with non-geographic predictors) | `ONLY_images/` generators and refit scripts; the non-geographic variant is not in the repository; aggregation not documented | See the paper |
| Table 9 (six-fold cross-validation, tabular and CNN) | Not documented | See the paper |
| Table 10 (ResNet18 with 3 km tiles and geographic predictors) | `Image_DNN/Neigh_3/` generators and refit scripts in `With_lat_long/` and `With_lat_long_osm14corine2000/`; aggregation not documented | See the paper |
| Table 11 (GLM and XGBoost with image embeddings) | `GLM_reg_with_ResNet18_emb.sh`, `XGB_with_ResNet18_emb.sh`, `NomicEmb_GLM_reg.sh`, `NomicEmb_XGB.sh`; aggregation not documented | See the paper |
| Table 12 (non-zero coefficients, regularized GLM, fold 3) | Not documented (`GLM_reg.py` prints the coefficients to the job log) | See the paper |
| Tables 13 to 15 (robustness over five experiments) | Not documented: four of the five experiments are not in the repository | See the paper |
| Figures 4 to 6 (histograms of claims, exposure and features) | Not documented | See the paper |
| Figures 7 and 8 (orthoimage tiles) | Not documented: the tiling code is not in the repository | See the paper |

## Citation

```bibtex
@article{alfonsosanchez2026geoclaim,
  title   = {Revealing Geography-Driven Signals in Zone-Level Claim Frequency Models: An Empirical Study using Environmental and Visual Predictors},
  author  = {Alfonso-S{\'a}nchez, Sherly and Bravo, Cristi{\'a}n and Stankova, Kristina G.},
  year    = {2026},
  journal = {arXiv preprint arXiv:2604.21893},
  doi     = {10.48550/arXiv.2604.21893},
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

The code is shared for the review of the paper only. All rights are reserved until the paper is
accepted, and the final licence will be published then. See [LICENSE](LICENSE).

The terms of the data are in `data/README.md`.

## Acknowledgements

The authors acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC) [Discovery Grants RGPIN-2020-07114 and RGPIN-2019-06586]. This work was funded, in part, by the Canada Research Chair Program [CRC-2024-00192] and was enabled, in part, by the support provided by Compute Ontario (https://www.computeontario.ca) and the Digital Research Alliance of Canada (https://alliancecan.ca).

Orthoimagery: © Institut géographique national @ Cartesius.be (for academic purposes).
