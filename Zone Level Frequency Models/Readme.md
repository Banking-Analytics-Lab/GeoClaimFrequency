# Script Descriptions

In this folder, the scripts for the models are devided in 3 groupps, namely:
1. Models with ONLY images: These includes the ones using for training only one scale of images, 0.5km, 1km, 3km or 5km, *MResNet18_extcv_single_config_ONLY_images.py* and the correctponding script for refitting using the best hyperparameters found *MResNet18_refit_images_ONLY.py*. And the model that uses all the radious scales at the same time, for training, *MResNet18_multiRadii_extcv_ONLY_images.py* and for refitting *MResNet18_multiRadii_ONLY_images_refit.py*.
2. Models with Only tabular data: This contains *GLM.py*, *GLM_reg.py*, *TabularDNN.py*, *XGB.py* and the script used for refitting the DNN model using the best set of hyperparameters, *TabularDNN_refit.py*.
3. Models with Tabular and Images: Given experiments on the model using one of the images scale and another tabular features, taraining and refiting stage given in *MResNet18_extcv_single_config.py* and *MResNet18_refit_single_config.py*
