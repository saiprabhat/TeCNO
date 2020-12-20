import pandas as pd
from torch.utils.data import Dataset
import pprint, pickle
from pathlib import Path
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from albumentations import (
    Compose,
    Resize,
    Normalize,
    ShiftScaleRotate,
)
import torch

root_dir = Path("/mnt/polyaxon/data1/MITI_CHE")
# Column of class labels
label_col = "class"
# Dataframe
df = {}
# Load the pickled dataframe
df["all"] = pd.read_pickle(root_dir / "dataframe/MITI_split_250px_25fps.pkl")

# get Video IDs from dataframe
video_ids = df["all"].video_idx.unique()
print(f'original video_ids: {video_ids}')
# shuffling the video indices for cross-validation
np.random.shuffle(video_ids)
print(f'shuffled video_ids: {video_ids}')
# Cross-validation; circular connectivity by specifying start index for training
# Block of constant size of 65 videos - 20 videos


# Define Train-Validation-Test video split
vids_for_training = video_ids[:65]
vids_for_val = video_ids[65:]






print(f'END OF TEST')