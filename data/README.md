# Data

This folder holds no data in git. The code reads its inputs from `$INPUT_DIR` (default `data`) and the image tiles from `$IMAGES_DIR` (default `data/Images_Ortho_95`). It writes the datasets it builds to `$WORK_DIR` (default `data/Data_creation`). The section "Paths" of the top-level `README.md` explains the variables.

## beMTPL97: the policy-level portfolio

- Source: the data set `beMTPL97` of the R package CASdatasets (Dutang and Charpentier), <https://dutangc.github.io/CASdatasets/>. It holds 163,212 policies with the variables `id`, `expo`, `claim`, `nclaims`, `amount`, `average`, `coverage`, `ageph`, `sex`, `bm`, `power`, `agec`, `fuel`, `use`, `fleet`, `postcode`, `long` and `lat`. The package gives the source as an unknown insurer.
- Licence: the `DESCRIPTION` file of CASdatasets (version 1.2-1) states `License: GPL (>= 2)`.
- Where the code expects it: `data/beMTPL97.csv`, a CSV export of the data frame with the column names above. `Preprocessing and Agreggation/Preprocessing_Belgian.ipynb` reads it as `../data/beMTPL97.csv`.

## Derived tables in this repository

- `Preprocessing and Agreggation/beMTPL97_filtered.csv` (14,980,022 bytes, 163,193 policies): beMTPL97 restricted to policies with at most three claims, written by `Preprocessing_Belgian.ipynb`. It is derived from beMTPL97, which CASdatasets distributes under GPL (>= 2).
- `Preprocessing and Agreggation/BeMTPL_Agg.csv` (186,726 bytes, 583 rows): the aggregation by postcode written by `creation_data_aggregated.py`. `Data_merge_all.sh` reads it from `$INPUT_DIR/BeMTPL_Agg.csv`.

## OpenStreetMap 2014

- Source: the Geofabrik extract of Belgium `belgium-140101.osm.pbf`, <https://download.geofabrik.de/europe/belgium.html>.
- Licence: Open Database License (ODbL), as cited in the paper.
- Where the code expects it: `$INPUT_DIR/OMS/belgium-140101.osm.pbf` (the folder is named `OMS` in the code).

## CORINE Land Cover 2000

- Source: Copernicus Land Monitoring Service, <https://land.copernicus.eu/en/products/corine-land-cover/clc-2000>. The code reads the raster `U2006_CLC2000_V2020_20u1.tif` (EPSG:3035, per a comment in the code).
- Licence: neither this repository nor the paper states it. See the product page.
- Where the code expects it: `$INPUT_DIR/OSM_2014_CORINE_2000/U2006_CLC2000_V2020_20u1.tif`.

## Orthoimagery 1995

- Source: the 1995 orthophotos of the Belgian National Geographic Institute (NGI), obtained through Cartesius.be, <https://www.cartesius.be>, for academic research.
- Terms, as stated in `Alternative Geo Data/Readme.md`: use for non-profit, non-commercial, teaching or scientific research purposes only, with the source and owner acknowledged as "© Institut géographique national @ Cartesius.be (for academic purposes)". The imagery is not in this repository.
- Where the code expects it: square tiles centred on each postcode's coordinates, at `$IMAGES_DIR/squares_R<r>km/Orth95_<lat>_<long>_R<r>.jpg`, with `<r>` one of `0.5`, `1` and `3`. The paper states that the tiles were cut with ArcGIS Pro and Python scripts. That code is not in this repository.

## Files built by the pipeline

- `unique_locations_BeMTPL97.csv` (columns `postcode`, `lat`, `long`): written by `Preprocessing and Agreggation/Extract_geo_location_options.ipynb`. The code reads it from `$INPUT_DIR/unique_locations_BeMTPL97.csv` and `$INPUT_DIR/OSM_2014_CORINE_2000/unique_locations_BeMTPL97.csv`.
- `osm_features_OSM2014_maskCLC2000_faster3.csv`: written by `Alternative Geo Data/osm14corine2000_feature_extract.py` in the current folder. `Data_merge_all.sh` reads it from `$INPUT_DIR/OSM_2014_CORINE_2000/`.
- `$WORK_DIR/With_osm14_corine2000/`, `$WORK_DIR/Stratified_division/`, `$WORK_DIR/Embeddings_nomic_v15/` and `$WORK_DIR/Embeddings_Resnet18/`: written or read by the scripts in `Data Splits by Experiment/`. The top-level `README.md` gives the order.

## Inputs that are not in the repository and whose origin is not documented

- `$INPUT_DIR/resnet18_weights.pth`: a state dictionary loaded into torchvision's `resnet18` before training the image models. Its source is not documented.
- `$INPUT_DIR/hyperparam_grid_ep15.csv`, `hyperparam_grid_ep30.csv` and `hyperparam_grid_ep60.csv`: the hyperparameter grids of the neural networks. The generated Slurm scripts skip one header line and read rows 0 to 7 as `CONFIG_IDX,BATCH_SIZE,HIDDEN_DIM,EPOCHS,LR,DROPOUT,OPTIMIZER,WEIGHT_DECAY,SEED,CONFIG_ID`. The grid values are not documented.

## Expected layout

```
data/
  beMTPL97.csv
  BeMTPL_Agg.csv
  unique_locations_BeMTPL97.csv
  resnet18_weights.pth
  hyperparam_grid_ep15.csv
  hyperparam_grid_ep30.csv
  hyperparam_grid_ep60.csv
  OMS/belgium-140101.osm.pbf
  OSM_2014_CORINE_2000/unique_locations_BeMTPL97.csv
  OSM_2014_CORINE_2000/U2006_CLC2000_V2020_20u1.tif
  OSM_2014_CORINE_2000/osm_features_OSM2014_maskCLC2000_faster3.csv
  Images_Ortho_95/squares_R0.5km/
  Images_Ortho_95/squares_R1km/
  Images_Ortho_95/squares_R3km/
  Data_creation/            (written by the pipeline)
```
