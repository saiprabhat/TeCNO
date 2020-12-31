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

class Cholec80FeatureExtract:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_mode = hparams.dataset_mode
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        self.fps_sampling = hparams.fps_sampling
        self.fps_sampling_test = hparams.fps_sampling_test
        self.miti_root_dir = Path(self.hparams.data_root +
                                    "/MITI_CHE")  # videos splitted in images
        self.transformations = self.__get_transformations()
        self.class_labels = [
            "PrePreparation",
            "Preparation",
            "Clipping",
            "Dissection",
            "Haemostasis1",
            "Retrieval",
            "Haemostasis2",
        ]

        # Weights obtained after Median Frequency Balancing
        weights = [
            1.6242660518999563,
            0.839382998657784,
            1.0,
            0.8745760497207751,
            1.9001433840697624,
            0.8167043332484509,
            1.4358846408551202,
        ]
        self.class_weights = np.asarray(weights)
        # Column of class labels
        self.label_col = "class"
        # Dataframe
        self.df = {}
        # Load the pickled dataframe
        self.df["all"] = pd.read_pickle(
            self.miti_root_dir / "dataframe/MITI_split_250px_25fps.pkl")

        
        # get Video IDs from dataframe
        self.video_ids = self.df["all"].video_idx.unique()
        print(f'original video_ids: {self.video_ids}')

        # Define Train-Validation-Test video split
        # self.vids_for_training = self.video_ids[:65]
        # self.vids_for_val = self.video_ids[65:]

        self.vids_for_training, self.vids_for_val, self.vids_for_test = \
                                        self.get_5fold_split(self.hparams.data_split, self.video_ids)
                                        
        print(f'#training video_ids: {len(self.vids_for_training)}\n{self.vids_for_training}\n')
        print(f'#validation video_ids: {len(self.vids_for_val)}\n{self.vids_for_val}\n')

        assert len(self.vids_for_training) == 52
        assert len(self.vids_for_val) == 13
        assert set(self.vids_for_training).issubset(set(self.video_ids)) == True
        assert set(self.vids_for_val).issubset(set(self.video_ids)) == True


        # Extract Train videos frames
        self.df["train"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_training)]
        # Extract Validation videos frames
        self.df["val"] = self.df["all"][self.df["all"]["video_idx"].isin(
            self.vids_for_val)]
        # Extract Test videos frames
        if hparams.test_extract:
            print(
                f"test extract enabled. Test will be used to extract the videos (testset = all)"
            )
            self.vids_for_test = video_ids
            self.df["test"] = self.df["all"]
        else:
            self.df["test"] = self.df["all"][self.df["all"]["video_idx"].isin(
                self.vids_for_test)]

        len_org = {
            "train": len(self.df["train"]),
            "val": len(self.df["val"]),
            "test": len(self.df["test"])
        }

        # Sub-sample train and validation set videos
        if self.fps_sampling < 25 and self.fps_sampling > 0:
            # sub-sampling factor
            factor = int(25 / self.fps_sampling)
            print(
                f"Subsampling(factor: {factor}) data: 25fps > {self.fps_sampling}fps"
            )
            # selecting sub-sampled frames from dataframe
            self.df["train"] = self.df["train"].iloc[::factor]
            self.df["val"] = self.df["val"].iloc[::factor]
            self.df["all"] = self.df["all"].iloc[::factor]
            # printing to console and verify sub-sampling
            for split in ["train", "val"]:
                print(
                    f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")
        
        # Sub-sample test set videos 
        if hparams.fps_sampling_test < 25 and self.fps_sampling_test > 0:
            # sub-sampling factor
            factor = int(25 / self.fps_sampling_test)
            print(
                f"Subsampling(factor: {factor}) data: 25fps > {self.fps_sampling}fps"
            )
            # selecting sub-sampled frames from dataframe
            self.df["test"] = self.df["test"].iloc[::factor]
            split = "test"
            # printing to console and verify sub-sampling
            print(f"{split:>7}: {len_org[split]:8} > {len(self.df[split])}")

        self.data = {}
        # Check if there is tool information
        if self.dataset_mode == "img_multilabel":
            for split in ["train", "val"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                    image_path_col="image_path",
                    add_label_cols=[
                        "tool_Grasper", "tool_Bipolar", "tool_Hook",
                        "tool_Scissors", "tool_Clipper", "tool_Irrigator",
                        "tool_SpecimenBag"
                    ])
            # Here we want to extract all features
            #self.df["test"] = self.df["all"].reset_index()
            self.df["test"] = self.df["test"].reset_index()
            self.data["test"] = Dataset_from_Dataframe(
                self.df["test"],
                self.transformations["test"],
                self.label_col,
                img_root=self.cholec_root_dir / "cholec_split_250px_25fps",
                image_path_col="image_path",
                add_label_cols=[
                    "video_idx", "image_path", "index", "tool_Grasper",
                    "tool_Bipolar", "tool_Hook", "tool_Scissors",
                    "tool_Clipper", "tool_Irrigator", "tool_SpecimenBag"
                ])
        
        # Multi-class classification of video frames
        if self.dataset_mode == "vid_multilabel":
            for split in ["train", "val", "test"]:
                self.df[split] = self.df[split].reset_index()
                self.data[split] = Dataset_from_Dataframe_video_based(
                    self.df[split],
                    self.transformations[split],
                    self.label_col,
                    img_root=self.miti_root_dir / "split_250px_250px",
                    image_path_col="image_path",
                )

    def __get_transformations(self):
        """
        Performs transformations on the `train`, `val` and `test` datasets.
        """
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1,
                             scale_limit=(-0.2, 0.5),
                             rotate_limit=15,
                             border_mode=0,
                             value=0,
                             p=0.7),
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations

    # def median_frequency_weights(
    #         self, file_list):  ## do only once and define weights in class
    #     frequency = [0, 0, 0, 0, 0, 0, 0]
    #     for i in file_list:
    #         frequency[int(i[1])] += 1
    #     median = np.median(frequency)
    #     weights = [median / j for j in frequency]
    #     return weights

    def get_5fold_split(self, split, video_ids):
        # LAST 20 VIDEOS FOR TEST
        vids_for_test = np.take(video_ids, np.arange(65, 85))
        # train = 52 videos and val = 13 videos
        if split == 1:
            inds_for_train = np.arange(0, 52)
            inds_for_val = np.arange(52, 65)
            vids_for_train = np.take(video_ids, inds_for_train)
            vids_for_val = np.take(video_ids, inds_for_val)

        elif split == 2:
            inds_for_train = np.arange(13, 65)
            inds_for_val = np.arange(0, 13)
            vids_for_train = np.take(video_ids, inds_for_train)
            vids_for_val = np.take(video_ids, inds_for_val)

        elif split == 3:
            inds_for_train = np.arange(26, (26 + 52))
            inds_for_train[inds_for_train > 64] = inds_for_train[inds_for_train > 64] - 65
            inds_for_val = np.arange((26 + 52), (26 + 52 + 13))
            inds_for_val[inds_for_val > 64] = inds_for_val[inds_for_val > 64] - 65
            vids_for_train = np.take(video_ids, inds_for_train)
            vids_for_val = np.take(video_ids, inds_for_val)

        elif split == 4:
            inds_for_train = np.arange(39, (39 + 52))
            inds_for_train[inds_for_train > 64] = inds_for_train[inds_for_train > 64] - 65
            inds_for_val = np.arange((39 + 52), (39 + 52 + 13))
            inds_for_val[inds_for_val > 64] = inds_for_val[inds_for_val > 64] - 65
            vids_for_train = np.take(video_ids, inds_for_train)
            vids_for_val = np.take(video_ids, inds_for_val)

        elif split == 5:
            nds_for_train = np.arange(52, (52 + 52))
            inds_for_train[inds_for_train > 64] = inds_for_train[inds_for_train > 64] - 65
            inds_for_val = np.arange((52 + 52), (52 + 52 + 13))
            inds_for_val[inds_for_val > 64] = inds_for_val[inds_for_val > 64] - 65
            vids_for_train = np.take(video_ids, inds_for_train)
            vids_for_val = np.take(video_ids, inds_for_val)
        else:
            raise NotImplementedError

        return vids_for_train, vids_for_val, vids_for_test

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 specific args options')
        cholec80_specific_args.add_argument("--fps_sampling",
                                            type=float,
                                            default=25)
        cholec80_specific_args.add_argument("--fps_sampling_test",
                                            type=float,
                                            default=25)
        cholec80_specific_args.add_argument(
            "--dataset_mode",
            default='video',
            choices=[
                'vid_multilabel', 'img', 'img_multilabel',
                'img_multilabel_feature_extract'
            ])
        cholec80_specific_args.add_argument("--test_extract",
                                            action="store_true")
        return parser


class Dataset_from_Dataframe_video_based(Dataset):
    """simple datagenerator from pandas dataframe"""

    # "video_id", "image_path", "time", "class"
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path"):
        # dataframe
        self.df = df
        # transforms
        self.transform = transform
        # "class"
        self.label_col = label_col
        # "image_path"
        self.image_path_col = image_path_col
        # root of split frames
        self.img_root = img_root
        # total number of frames in the split
        self.number_frames = len(self.df[self.image_path_col])
        # path to all frames in the split
        self.allImages = self.df[self.image_path_col].tolist()
        # labels of all frames in the split
        self.allLabels = self.df[self.label_col].tolist()

    def __len__(self):
        return self.number_frames

    def __getitem__(self, index):

        p = self.img_root / self.allImages[index]
        im = np.asarray(Image.open(p), dtype=np.uint8)
        # performing the transform
        if self.transform:
            final_im = self.transform(image=im)["image"]            

        # label for this frame
        label = torch.tensor(self.allLabels[index], dtype=torch.long)

        return final_im, label


class Dataset_from_Dataframe(Dataset):
    def __init__(self,
                 df,
                 transform,
                 label_col,
                 img_root="",
                 image_path_col="path",
                 add_label_cols=[]):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root
        self.add_label_cols = add_label_cols

    def __len__(self):
        return len(self.df)

    def load_from_path(self, index):
        img_path_df = self.df.loc[index, self.image_path_col]
        p = self.img_root / img_path_df
        X = Image.open(p)
        X_array = np.array(X)
        return X_array, p

    def __getitem__(self, index):
        X_array, p = self.load_from_path(index)
        if self.transform:
            X = self.transform(image=X_array)["image"]
        label = torch.tensor(int(self.df[self.label_col][index]))
        add_label = []
        for add_l in self.add_label_cols:
            add_label.append(self.df[add_l][index])
        X = X.type(torch.FloatTensor)
        return X, label, add_label

