# DCASE-TASK1 2020

Code for running DCASE TASK1 experiments (acoustic scene classification).

How to use the code:

1. Run `melspects.py` to extract features. Script accepts the location of the extracted DCASE data (`<data>`) and a path to store the exracted features in a folder (`<features>`)
2. Run `training.py` to train a model. Script accepts `<data>` as its `--data-root` parameter, `<features>/features.csv` as its `--features` parameter and a `--results-root` to store results in
3. Run `benchrunner.py` to start a grid search for the training where training is performed over different combinations of parameters such as (`--learning-rate`, `--batch-size` and so on).

If you want to include new models, adapt the `--approach` parameter to support them.

Download data from the official website `https://dcase.community/challenge2020/index`
