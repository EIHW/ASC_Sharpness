# Minima Sharpness for Acoustic Scene Classification

This repository contains two subfolders, merging repositories for model training and the calculation of sharpness values for a given minimum:
- `training/`: Training of the DCASE2020-Task1 for Acoustic Scene classification based on Mel-spectrograms and amongst others the PANNs models CNN10 and CNN14. Training code by **Andreas Triantafyllopoulos**, extended grid search by **Simon David Noel Rampp**.  
- `sharpness/`: Calculcation of sharpness values for loss function minima (e.g., models trained with the `training/` folder) based on filter-normalised loss visualisation and $\epsilon$-sharpness. Original Code from [Visualizing the Loss Landscape of Neural Nets](https://github.com/tomgoldstein/loss-landscape) and extended by **Manuel Milling**.

For reproduction of the paper either train models (`training/`) or download [pretrained models](https://zenodo.org/record/8335153) and calculate the sharpness values by following instructions in `sharpness/`.

Results are summarized in `ICASSP Sharpness Experiment Results - Cut_ICASSP_Version.csv`.

Paper for this code is under under review. 
