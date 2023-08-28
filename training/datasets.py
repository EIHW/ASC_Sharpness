import audiofile
import numpy as np
import pandas as pd
import torch
from os.path import basename
import torchaudio
from transformers import ASTFeatureExtractor

class CachedDataset(torch.utils.data.Dataset):
    r"""Dataset of cached features.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
        self, 
        df: pd.DataFrame,
        features: pd.DataFrame,
        target_column: str, 
        transform=None, 
        target_transform=None,
        feature_dir=""
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, item):
    #     index = self.indices[item]
    #     signal = np.load(self.features.loc[index, 'features'] + '.npy')
    #     target = self.df[self.target_column].loc[index]
    #     if isinstance(self.target_column, list) and len(self.target_column) > 1:
    #         target = np.array(target.values)

    #     if self.transform is not None:
    #         signal = self.transform(signal)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return signal, target

    def __getitem__(self, item):
        index = self.indices[item]
        if self.feature_dir == "":
            signal = np.load(self.features.loc[index, 'features'] + '.npy')
        else:
            signal = np.load(self.feature_dir + basename(self.features.loc[index, 'features'] + '.npy'))
        target = self.df[self.target_column].loc[index]
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if self.transform is not None:
            # print(signal.shape)
            signal = self.transform(signal)
            # print(signal.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target


class WavDataset(torch.utils.data.Dataset):
    r"""Dataset of raw audio data.

    Args:
        df: partition dataframe containing labels
        features: dataframe with paths to features
        target_column: column to find labels in (in df)
        transform: function used to process features
        target_transform: function used to process labels
    """

    def __init__(
        self, 
        df: pd.DataFrame,
        features: pd.DataFrame,
        target_column: str, 
        transform=None, 
        target_transform=None,
    ):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.indices = list(self.df.index)
        self.resample = torchaudio.transforms.Resample(44100, 16000)
        self.feature_extractor = ASTFeatureExtractor()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        index = self.indices[item]
        # signal = audiofile.read(index, always_2d=False)[0]
        signal, fs = audiofile.read(index, always_2d=True)
        target = self.df[self.target_column].loc[index]
        if isinstance(self.target_column, list) and len(self.target_column) > 1:
            target = np.array(target.values)

        if fs == 44100:
            signal = self.resample(torch.from_numpy(signal)).numpy()
        signal = self.feature_extractor(
            signal, 
            sampling_rate=16000,
            padding="max_length", 
            return_tensors="pt"
        ).input_values[0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target