import json
from typing import Any
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from annotator.canny import CannyDetector
from annotator.util import HWC3


from myutils import Degrade


class CannyDataset(Dataset):

    def __init__(
        self,
        min_resize_res: int = 512,
        max_resize_res: int = 512,
        crop_res: int = 512,
        low_threshold: int = 100,
        high_threshold: int = 200,
        path_seed: str = 'srcs/seeds_openimage.txt',
    ):
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res

        self.apply_canny = CannyDetector()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        with open(path_seed, 'rt') as f:
            self.data = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        # LOAD IMAGE
        path = self.data[i]
        x_clean = Image.open(path).convert('RGB')

        # FIX PROMPT
        prompt = "a high-quality, detailed, and professional image"

        # IMAGE RESIZE AND CROP
        w, h = x_clean.size
        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1,
                                  ()).item()

        if w > h:
            x_clean = x_clean.resize((int(w / h * reize_res), reize_res),
                                     Image.Resampling.LANCZOS)
        else:
            x_clean = x_clean.resize((reize_res, int(h / w * reize_res)),
                                     Image.Resampling.LANCZOS)
        # CROP
        x_clean = torchvision.transforms.RandomCrop(self.crop_res)(x_clean)

        # PROCESS
        x_cond = self.apply_canny(x_clean, self.low_threshold,
                                    self.high_threshold)

        # Normalize source images to [0, 1].
        x_cond = np.array(x_cond) / 255.0

        # Normalize target images to [-1, 1].
        x_clean = 2.0 * np.array(x_clean) / 255.0 - 1

        return dict(jpg=x_clean, txt=prompt, hint=x_cond)
