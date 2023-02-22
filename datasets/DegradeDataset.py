import json
from typing import Any
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from myutils import Degrade


class DegradeDataset(Dataset):

    def __init__(
        self,
        min_resize_res: int = 512,
        max_resize_res: int = 512,
        crop_res: int = 512,
        path_seed: str = 'srcs/seeds_openimage.txt',
        targets=["blur", "gray", "noise", "downsample"],
    ):
        print("TARGET DEGRADATIONS:", targets)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.degrader = Degrade(targets=targets)

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
        x_deg, _ = self.degrader(x_clean)

        # Normalize source images to [0, 1].
        x_deg = np.array(x_deg) / 255.0

        # Normalize target images to [-1, 1].
        x_clean = 2.0 * np.array(x_clean) / 255.0 - 1

        return dict(jpg=x_clean, txt=prompt, hint=x_deg)
