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
def test():
    root_dir = Path("/mnt/polyaxon/data1/MITI_CHE")
    # Column of class labels
    label_col = "class"
    # Dataframe
    df = {}
    # Load the pickled dataframe
    df["all"] = pd.read_pickle(root_dir / "dataframe/MITI_split_250px_25fps.pkl")

    # get Video IDs from dataframe
    video_ids = df["all"]["video_idx"].unique()
    print(f'original video_ids: {video_ids}')
    # shuffling the video indices for cross-validation
    np.random.shuffle(video_ids)
    print(f'shuffled video_ids: {video_ids}')
    # Cross-validation; circular connectivity by specifying start index for training
    # Block of constant size of 65 videos - 20 videos

    # Median-frequency-balancing for class-imbalance
    classes = df["all"]["class"]
    frequency = [0, 0, 0, 0, 0, 0, 0]
    for i in classes:
        frequency[int(i)] += 1
    median = np.median(frequency)
    weights = [median / j for j in frequency]
    print()
    print(f"weights: {weights}")
    # Define Train-Validation-Test video split
    vids_for_training = video_ids[:65]
    vids_for_val = video_ids[65:]
    # Extract Train videos frames
    df["train"] = df["all"][df["all"]["video_idx"].isin(vids_for_training)]
    # Extract Validation videos frames
    df["val"] = df["all"][df["all"]["video_idx"].isin(vids_for_val)]
    # Extract Test videos frames
    print(f"test extract enabled. Test will be used to extract the videos (testset = all)")
    vids_for_test = video_ids
    df["test"] = df["all"]


    len_org = {
                "train": len(df["train"]),
                "val": len(df["val"]),
                "test": len(df["test"])
            }

    fps_sampling = 1
    fps_sampling_test = 1
    # Sub-sample train and validation set videos
    if fps_sampling < 25 and fps_sampling > 0:
        # sub-sampling factor
        factor = int(25 / fps_sampling)
        print(
            f"Subsampling(factor: {factor}) data: 25fps > {fps_sampling}fps"
        )
        # selecting sub-sampled frames from dataframe
        df["train"] = df["train"].iloc[::factor]
        df["val"] = df["val"].iloc[::factor]
        df["all"] = df["all"].iloc[::factor]
        # printing to console and verify sub-sampling
        for split in ["train", "val"]:
            print(
                f"{split:>7}: {len_org[split]:8} > {len(df[split])}")

    # Sub-sample test set videos 
    if fps_sampling_test < 25 and fps_sampling_test > 0:
        # sub-sampling factor
        factor = int(25 / fps_sampling_test)
        print(
            f"Subsampling(factor: {factor}) data: 25fps > {fps_sampling}fps"
        )
        # selecting sub-sampled frames from dataframe
        df["test"] = df["test"].iloc[::factor]
        split = "test"
        # printing to console and verify sub-sampling
        print(f"{split:>7}: {len_org[split]:8} > {len(df[split])}")

    data = {}

    # Multi-class classification of video frames
    for split in ["train", "val", "test"]:
        df[split] = df[split].reset_index()
        data[split] = Dataset_from_Dataframe_video_based(
            df[split],
            transform=False,
            label_col="class",
            img_root=root_dir / "split_250px_250px",
            image_path_col="image_path",
        )
    print(f'END OF TEST')

class Dataset_from_Dataframe_video_based(Dataset):
    """simple datagenerator from pandas dataframe"""

    # "image_path"", "class", "time", "video", "tool_Grasper", "tool_Bipolar", "tool_Hook", "tool_Scissors", "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
    # "video_id", "image_path", "time", "class"
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path"):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root
        self.starting_idx = self.df["video_idx"].min()
        print(self.starting_idx)

    def __len__(self):
        return len(self.df.video_idx.unique())

    def __getitem__(self, index):
        # starting index
        sindex = self.starting_idx + index
        # select all frames for a given video
        img_list = self.df.loc[self.df["video_idx"] == sindex]
        # 
        videos_x = torch.zeros([len(img_list), 3, 224, 224], dtype=torch.float)
        label = torch.tensor(img_list[self.label_col].tolist(),
                             dtype=torch.int)
        f_video = self.load_cholec_video(img_list)
        if self.transform:
            for i in range(len(f_video)):
                videos_x[i] = self.transform(image=f_video[i],
                                             mask=None)["image"]
        
        #print(f"Index of video: {index} - sindex: {sindex} - len_label: {label.shape[0]} - len_vid: {videos_x.shape[0]}")
        assert videos_x.shape[0] == label.shape[0], f"weird shapes at {sindex}"
        assert videos_x.shape[
            0] > 0, f"no video returned shape: {videos_x.shape[0]}"
        return videos_x, label

    def load_cholec_video(self, img_list):
        f_video = []
        allImage = img_list[self.image_path_col].tolist()
        for i in range(img_list.shape[0]):
            p = self.img_root / allImage[i]
            im = Image.open(p)
            f_video.append(np.asarray(im, dtype=np.uint8))
        f_video = np.asarray(f_video)
        return f_video

if __name__ == "__main__":
    test()