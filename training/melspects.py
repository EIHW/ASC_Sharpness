import argparse
import audiofile as af
import glob
import numpy as np
import os
import pandas as pd
import torch
import tqdm
import torchaudio
import torchlibrosa


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Mel-spectrogram feature extraction')
    parser.add_argument(
        'root',
        help='Path to unzipped DCASE-Task1 data'
    )
    parser.add_argument(
        'dest',
        help='Path to store features in'
    )
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    ref = 1.0
    amin = 1e-10
    top_db = None

    spectrogram = torchlibrosa.stft.Spectrogram(
        n_fft=512, 
        win_length=512, 
        hop_length=160
    )
    mel = torchlibrosa.stft.LogmelFilterBank(
        sr=16000, 
        fmin=50, 
        fmax=8000, 
        n_mels=64, 
        n_fft=512, 
        ref=ref, 
        amin=amin, 
        top_db=top_db
    )
    transform = lambda x: mel(spectrogram(x)).squeeze(1)
    
    filenames = []
    df = pd.read_csv(os.path.join(args.root, 'meta.csv'), sep='\t')
    for counter, row in tqdm.tqdm(enumerate(df.iterrows()), total=len(df), desc='Melspects'):
        audio, fs = af.read(
            os.path.join(args.root, row[1]['filename']),
            always_2d=True
        )
        if fs != 16000:
            audio = torchaudio.transforms.Resample(fs, 16000)(torch.from_numpy(audio))
        else:
            audio = torch.from_numpy(audio)
        logmel = transform(audio)
        filename = os.path.join(args.dest, '{:012}'.format(counter))
        np.save(filename, logmel)
        filenames.append(filename)
    df.set_index('filename', inplace=True)
    features = pd.DataFrame(
        data=filenames,
        index=df.index,
        columns=['features']
    )
    features.reset_index().to_csv(os.path.join(args.dest, 'features.csv'), index=False)