# Script Descriptions

This folder contains the scripts used to generate all augmented datasets.

## 1. Data_merge.py
This script merges the `osm14_corine2000` features with the tabular variables.  
It is called within the script located in the `With_osm14_corine2000` folder.

## 2. Data_creation Folder
The `Data_creation` folder provides an example of one of the five experiments conducted in our research.  
Each experiment is dependent on a random seed and contains the following subfolders:

### Stratified_division
This subfolder contains the computation of the 6 stratified data folds using random seed **611**.  
The process is implemented in the script `Division_6.py`.

### Augmented_data_with_embeddings
This step computes embeddings using the selected **ResNet18** model.

- The model is chosen based on the lowest RMSE obtained during the 5-fold cross-validation stage, using **only image data**.
- This stage depends on the outer fold (`outerfoldx`).
- The script used is `Augmented_data_by_folds.py`.

In this example, only `outerfold0` and `outerfold1` are shown, but the full experiment includes folds from `outerfold0` to `outerfold5`.

### Embeddings_nomic_v15
This stage augments the dataset using the **nomic v1.5** model.

It requires:
- `Saving_nomic_v15_weights.py`: Saves the model weights locally.
- `creation_nomic3km_emb.py`: Generates the embeddings using the nomic v1.5 model.

## Notes
Only one `Data_creation` folder is included here for demonstration purposes.  
To avoid redundancy, the folders `Data_creation_1` to `Data_creation_4` have been omitted.
