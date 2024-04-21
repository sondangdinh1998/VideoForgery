import os
import glob
import random

import cv2
from albumentations import Compose as ICompose
from hydra.utils import instantiate

import torch
from torch.utils.data import Dataset

from .augment import Compose as VCompose


class DeepfakeDataset(Dataset):
    def __init__(self, dirpath, augment=None):
        super().__init__()
        self.num_frames = 32

        num_frames = range(self.num_frames)
        additional_targets_keys = ["image" + str(i) for i in num_frames]
        additional_targets_values = ["image" for i in num_frames]
        self.additional_targets = dict(
            zip(additional_targets_keys, additional_targets_values)
        )

        mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
        std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.mean = mean.reshape(1, 3, 1, 1).contiguous()
        self.std = std.reshape(1, 3, 1, 1).contiguous()

        self.dataset = self._load_dataset(dirpath)
        self.image_augment, self.video_augment = self._load_augment(augment)

    def __getitem__(self, index):
        video_filepath = self.dataset[index]

        image_dirpath = video_filepath.replace("/videos", "/boxes")
        image_dirpath = image_dirpath.replace(".mp4", os.path.sep)

        image_filepaths = sorted(os.listdir(image_dirpath))
        image_filepaths = [
            os.path.join(image_dirpath, path) for path in image_filepaths
        ]

        images = self._get_clip(image_filepaths)

        if self.image_augment is not None:
            arguments = {}
            for i, key in enumerate(self.additional_targets):
                key = key if key != "image0" else "image"
                arguments[key] = images[i]

            images = self.image_augment(**arguments)
            images = [images[key] for key in images]

        if self.video_augment is not None:
            images = self.video_augment(images)

        images = [torch.from_numpy(img).permute(2, 0, 1) for img in images]
        images = torch.stack(images, dim=0).sub(self.mean).div(self.std)

        target = torch.tensor(0 if "original" in video_filepath else 1)

        return images, target

    def __len__(self):
        return len(self.dataset)

    def _load_dataset(self, dirpath):
        dataset = glob.glob(f"{dirpath}/**/*.mp4", recursive=True)
        return dataset

    def _load_augment(self, augment_config):
        if augment_config is None:
            return None, None

        image_augment = augment_config.get("image", None)
        if image_augment is not None:
            image_augment = ICompose(
                [instantiate(aug) for _, aug in image_augment.items()],
                additional_targets=self.additional_targets,
            )

        video_augment = augment_config.get("video", None)
        if video_augment is not None:
            video_augment = VCompose(
                [instantiate(aug) for _, aug in video_augment.items()],
            )

        return image_augment, video_augment

    def _get_clip(self, filepaths):
        start_idx = random.randint(0, len(filepaths) - self.num_frames - 1)
        filepaths = filepaths[start_idx: start_idx + self.num_frames].copy()

        images = [cv2.imread(filepath) for filepath in filepaths]
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        return images
