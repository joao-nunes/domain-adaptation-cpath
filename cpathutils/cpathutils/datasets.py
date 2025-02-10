from typing import Any
from torch.utils.data import Dataset
from openslide import *
from skimage.filters import gaussian as gaussian_blur
import torch.utils.data as data
import numpy as np
import csv
import random
import albumentations as A
from albumentations.augmentations.transforms import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import torch
import sys
import json
import numbers
import os
import cv2

from skimage import color
from copy import deepcopy
import pandas as pd
from transformers import  AutoImageProcessor

from tiatoolbox.tools.stainnorm import VahadaneNormalizer

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fnmatch import fnmatch

import pandas as pd

from copy import deepcopy

SEED=42
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"



def fourier_amplitude_mixup(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [H, W, C]
    Source: https://github.com/MediaBrain-SJTU/FACT/blob/a877cc86acc4d29fb7589c8ac571c8aef09e5fd8/data/data_utils.py#L84
    """
    lam = np.random.uniform(0, alpha)

    assert img1.shape == img2.shape
    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img12 = img2_abs * (np.e ** (1j * img2_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img12 = np.real(np.fft.ifft2(img12, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img12 = np.uint8(np.clip(img12, 0, 255))

    return img21, img12


class BaseDataset(Dataset):

    def __init__(
        self,
        jsonfile: str = "./data/train_test_splits/test.json",
        libraryfile: str = "./data/train_otsu_512_100_map.tar",
        augment=True,
        transform: Any = None,
        norm: bool = False,
        size: int = 512,
        M: int = 2,
        multiview: bool = False,
        imp=True,
        ):
            
            super(BaseDataset, self).__init__()

            if transform is None and augment:
                
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.125, 0.125),
                            saturation=(0.875, 1.125),
                            brightness=(0.875, 1.2),
                            contrast=0.2,
                            p=0.5,
                        ),
                        HEDJitter(betas=(0.1, 0.0075), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.35),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.35),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                
            elif transform is not None:
                self.transform = transform
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ])
            self.transform_w = A.Compose([
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                ToTensorV2(),
                ])

            with open(jsonfile) as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.slideIDX = data["slideIDX"]
                self.targets = data["targets"]
            self.augment = augment
            lib = torch.load(libraryfile)
            
            print('Number of tiles: {}'.format(len(self.grid)))
            self.slidenames = lib['slides']
            self.mode = 1
            self.mult = lib['mult']
            self.targets2 = lib['slide_target']
            self.size = int(np.round(size*lib['mult']))
            self.level = lib['level']
            
            self.slideIDX = np.array(self.slideIDX)
            self.grid = np.array(self.grid)
            
            self.targets = np.array(self.targets)
            self.size = int(np.round(size*lib['mult']))
        
            self.level = lib['level']
            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.0),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ])

            self.random_scale_rotate = A.Compose([
                A.Rotate((-90, 90), p=0.5),
                A.RandomScale(scale_limit=(-0.2, 1.2), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ])
            self.M = M
            if norm:
                normalizer = VahadaneNormalizer()
                norm = []
                for f in os.listdir("./data/stain_norm_targets"):
                    f = os.path.join(
                        "./data/stain_norm_targets",
                        f
                    )
                    target = Image.open(f)
                    target = np.array(target)
                    norm.append(normalizer.fit(target))
            else:
                norm = None
            self.norm = norm
            self.multiview = multiview
            self.imp = imp
            
    def __getitem__(self,index):
        
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]
        if self.imp:
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
            except:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/").replace("non_annotated","extra_500"))
        else:
            slide = open_slide(self.slidenames[slideIDX])

        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img = np.array(img)
            aug = self.transform_w(image=img)
            img_w = aug["image"]
            if self.M == 1:
                img = self.rotate(image=img)["image"]
            else:
                img = self.random_scale_rotate(image=img)["image"]
        else:
            img = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img = np.array(img)
            aug = self.transform_w(image=img)
            img_w = aug["image"]

        if self.norm is not None:
            j = np.random.choice(np.arange(len(self.norm)))
            img = self.norm[j].transform(img)
        
        if index < len(self.grid)-1:
            next_slideIDX = self.slideIDX[index+1]
            if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                slide.close()
                del slide
        else:
            slide.close()
            del slide

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.multiview:
            return img_w, img
        return img, self.targets[index]

    def __len__(self):
        return len(self.grid)


class TilesDataset(Dataset):

    def __init__(
        self,
        root="/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/",
        path="CRC/tiles-annot-567-train/level-0",
        augment=False,
        transform: Any = None,
        slide_consistent_transform: bool = False,
        transform_params_path: str = "/nas-ctm01/homes/jdfernandes/datasets/slide_level_augs/",
        pattern="*.png",
        size: int = 512,
        M: int = 2,
        multiview: bool = False,
        N: int = 2,
        cons_reg: bool = False,
        return_key: bool = False,
        is_vit=False,
        processor: Any = AutoImageProcessor.from_pretrained("owkin/phikon"),
        return_pil_image: bool = False,
        vit: str = "phikon",
        transform_chunk_size: int = 5000,
        transform_fold: int = 0,
        n_transforms: int = 20,
        ):
            
            super(TilesDataset, self).__init__()
            
            
            self.image_ids = []           
            for subdir in os.listdir(os.path.join(root, path)):
                files = os.listdir(os.path.join(root, path, subdir))
                files.sort()
                for file in files:
                    self.image_ids.append(os.path.join(root, path, subdir, file))
            
            self.slide_consistent_transform = slide_consistent_transform

            if transform is None and augment and not self.slide_consistent_transform:
                
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.125, 0.125),
                            saturation=(0.875, 1.125),
                            brightness=(0.875, 1.2),
                            contrast=0.2,
                            p=0.5,
                        ),
                        HEDJitter(betas=(0.1, 0.0075), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.35),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.35),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                
            elif transform is not None:
                self.transform = transform
                
            elif self.slide_consistent_transform:
            
                self.transform = None
                self.transform_params_path = transform_params_path
                with open(os.path.join(self.transform_params_path, "slide_transforms.json"), "r") as f:
                    self.transform_params = json.load(f)

                image_ids = []
                for j in range(n_transforms):
                    for img_id in self.image_ids:

                        img_id = (".").join([
                        img_id.split(".")[0]+"-"+str(j),
                        img_id.split(".")[-1]
                        ])

                        image_ids.append(img_id)
                 
                self.image_ids = image_ids
                self.transform_chunk_size=transform_chunk_size
                self.transform_fold = transform_fold
                    
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ])
            self.transform_w = A.Compose([
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                ToTensorV2(),
                ])

            
            self.augment = augment
            self.multiview = multiview
            
            
            if os.path.exists(os.path.join(root, os.path.dirname(path), "targets.csv")):
                with open(os.path.join(root, os.path.dirname(path), "targets.csv"), "r") as f:
                    df = pd.read_csv(f)
                self.targets = {k: v for k,v in zip(df["img_id"].tolist(), df["target"].tolist())}
            else:
                self.targets = None
            
            self.cons_reg = cons_reg

            self.size = size        
            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.0),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ])

            self.random_scale_rotate = A.Compose([
                A.Rotate((-90, 90), p=0.5),
                A.RandomScale(scale_limit=(-0.2, 1.2), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ])
            self.M = M
            self.transform_w = A.Compose([
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                ToTensorV2(),
                ])
            self.return_key = return_key
            self.is_vit = is_vit
            self.image_processor = processor
            self.return_pil_image = return_pil_image
            self.N = N
            self.coords = []
            for image_id in self.image_ids:
                coords = image_id.split(".")[0].split("-")
                coords = (int(coords[-2]), int(coords[-1]))
                self.coords.append(coords)
            self.vit=vit
            
            
    def __getitem__(self,index):
        
        if not self.slide_consistent_transform:
                img = Image.open(self.image_ids[index]).convert("RGB")
                key = self.image_ids[index].split(".")[0].split("/")[-1]
                k = ("-").join(key.split("-")[:-2])
        else:
                array = self.image_ids[index].split(".")[0].split("/")[-1].split("-")
                del array[-3:-1]
                k = ("-").join(array)
                
                img_id = (".").join([
                    ("-").join(self.image_ids[index].split(".")[0].split("-")[:-1]),
                    self.image_ids[index].split(".")[-1]
                ])
                key = img_id.split(".")[0].split("/")[-1]
                
                img = Image.open(img_id).convert("RGB")
        img = np.array(img)
        if self.N != 1:
            
            coords = self.image_ids[index].split(".")[0].split("-")
            coords = (int(coords[-2]), int(coords[-1]))
            
            ref_coords = [
            (coords[0] - self.size, coords[1] - self.size),
            (coords[0] - self.size, coords[1]),
            (coords[0] - self.size, coords[1] + self.size),
            (coords[0], coords[1] - self.size),
            (coords[0], coords[1] + self.size),
            (coords[0] + self.size, coords[1] - self.size),
            (coords[0] + self.size, coords[1]),
            (coords[0] + self.size, coords[1] + self.size)]
            # remove non-existing coords due to otsu thresholding
            ref_coords = [coord for coord in ref_coords if coord in self.coords]
            if len(ref_coords) != 0:
                coord = random.choice(ref_coords)
                imgpath = self.image_ids[index].split(".")[0].split("-")
                imgpath[-2] = str(coord[0])
                imgpath[-1] = str(coord[1])
                imgpath = ("-").join(imgpath)+".png"
                view2 = Image.open(imgpath).convert("RGB")
            if not self.is_vit:
                view2 = np.array(view2)
            else:
                view2 = deepcopy(img)
         
        if self.return_pil_image:
            if self.return_key:
                return k, img
            return img
        
        if (random.random() < 0.5 and self.augment and not self.is_vit and not self.slide_consistent_transform):
            if self.M == 1:
                img = self.rotate(image=img)["image"]
            else:
                img = self.random_scale_rotate(image=img)["image"]
                if self.N != 1:
                    view2 = self.random_scale_rotate(image=view2)["image"]

        if self.transform is not None:
            img_s = self.transform(image=img)["image"]
            if self.is_vit:
                img = Image.fromarray(img_s, mode="RGB")
            if self.N != 1:
                view2 = self.transform(image=view2)["image"]
        else:
            
            transform_index = self.transform_params[k]
            
            idx = transform_index // self.transform_chunk_size
            file_id = "slide_level_aug_params_chunk_"+str(idx)+"_fold_"+str(self.transform_fold)+".json"
            transform_params_file = os.path.join(self.transform_params_path, file_id)
            with open(transform_params_file, "r") as f:
                data = json.load(f)[transform_index % self.transform_chunk_size]
            img_s = A.ReplayCompose.replay(data["replay"], image=img)

        if self.is_vit:

            inputs = self.image_processor(img, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].squeeze()
                    
            if self.return_key:
                if self.targets is not None:
                    target = torch.tensor(self.targets[key], dtype=torch.int64)
                    return k, inputs, target
                return k, inputs
            else:
                if self.targets is not None:
                    target = torch.tensor(self.targets[key], dtype=torch.int64)
                    return inputs, target
                return inputs
                        
        if self.multiview:
            img_w = self.transform(image=img)["image"]
        
        if self.cons_reg:
            img_w = self.transform_w(image=img)["image"]
        
        if self.targets is not None:
            target = torch.tensor(self.targets[key], dtype=torch.int64)
            
            if self.multiview or self.cons_reg:
                if self.return_key:
                    return k, img_w, img_s, target
                if self.N != 1:
                    target2 = torch.tensor(self.targets[imgpath.split(".")[0].split("/")[-1]], dtype=torch.int64)
                    return img_w, img_s, view2, target, target2
                return img_w, img_s, target
            if self.return_key:
                return k, img_s, target
            return img_s, target
        else:
            if self.multiview or self.cons_reg:
                if self.return_key:
                    return k, img_w, img_s
                return img_w, img_s
            else:
                if self.return_key:
                    return k, img_s
            return img_s
                

    def __len__(self):
        return len(self.image_ids)

class IntraInstanceDataset(Dataset):

    def __init__(
        self,
        jsonfile: str = "./data/train_split.json",
        libraryfile: str = "./data/train_otsu_512_100_map.tar",
        transform: Any = None,
        size: int = 512,
        augment: bool = True,
        M: int = 1,
        ):

            super(IntraInstanceDataset, self).__init__()
            assert M in [1, 2], "M should be either 1: 'weak' or 2: 'mild'"
            if transform is None and augment:
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.125, 0.125),
                            saturation=(0.875, 1.125),
                            brightness=(0.875, 1.2),
                            contrast=0.2,
                            p=0.5,
                        ),
                        HEDJitter(betas=(0.1, 0.0075), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.2),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
 
            elif transform is not None and augment:
                self.transform = transform
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ])

            with open(jsonfile) as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.slideIDX = data["slideIDX"]
                self.targets = data["targets"]
           
            lib = torch.load(libraryfile)
            
            print('Number of tiles: {}'.format(len(self.grid)))
            self.slidenames = lib['slides']
            self.mode = 1
            self.mult = lib['mult']
            self.targets2 =lib['slide_target']
            self.size = int(np.round(size*lib['mult']))
            self.level = lib['level']
            
            self.slideIDX = np.array(self.slideIDX)
            self.grid = np.array(self.grid)
            
            self.targets = np.array(self.targets)
            print(lib['level'])
            self.level = lib['level']
            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.0),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ])

            self.random_scale_rotate = A.Compose([
                A.Rotate((-90, 90), p=0.5),
                A.RandomScale(scale_limit=(-0.15, 1.15), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ])
            self.M = M
            self.augment = augment

    def __getitem__(self,index):
        
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]

        slide_ids = list(np.where(np.array(self.slideIDX)==slideIDX)[0])
        # ref_coords: coord of all neibhouring tiles (8 connectivity)
        ref_coords = [
            (coord[0] - self.size, coord[1] - self.size),
            (coord[0] - self.size, coord[1]),
            (coord[0] - self.size, coord[1] + self.size),
            (coord[0], coord[1] - self.size),
            (coord[0], coord[1] + self.size),
            (coord[0] + self.size, coord[1] - self.size),
            (coord[0] + self.size, coord[1]),
            (coord[0] + self.size, coord[1] + self.size)]
        
        # remove from ref_coords regions where there is a lot of slide background
        ctx_coords = [i for i in slide_ids if (self.grid[i][0], self.grid[i][1]) in ref_coords]
        
        # ensure only tiles completely filled with tisue are used
        # if no neighbour is eligible (e.g., due to Otsu thresholding in S in HSV colour space)
        # return the tile itself
       
        if len(ctx_coords) == 0:
            j = index
        else:
            j = random.choice(ctx_coords)
        ctx_coord = self.grid[j]
        
        try:
            slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
        except:
            slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/").replace("non_annotated","extra_500"))
        
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img = np.array(img)
            if self.M == 1:
                img = self.rotate(image=img)["image"]
            else:
                img = self.random_scale_rotate(image=img)["image"]
        else:
            img = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img = np.array(img)
        
        if (random.random() < 0.5 and self.augment):
            ctx_coord = [ctx_coord[0]-256, ctx_coord[1]-256]
            ctx = slide.read_region(ctx_coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            ctx = np.array(ctx)
            if self.M == 1:
                ctx = self.rotate(image=ctx)["image"]
            else:
                ctx = self.random_scale_rotate(image=ctx)["image"] 
        else:
            ctx = slide.read_region(ctx_coord, self.level,(self.size, self.size)).convert('RGB')
            ctx = np.array(ctx)
      
        if index < len(self.grid)-1:
            next_slideIDX = self.slideIDX[index+1]
            if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                slide.close()
                del slide
        else:
            slide.close()
            del slide
        if self.mult != 1:
            img = img.resize((224, 224),Image.BILINEAR)
            ctx = ctx.resize((224, 224),Image.BILINEAR)

        img = self.transform(image=img)["image"]
        ctx = self.transform(image=ctx)["image"]
  
        return img, ctx, self.targets[index], self.targets[j]
    
    def __len__(self):
        return len(self.grid)

class IntraInstanceCausalDataset(Dataset):

    def __init__(
        self,
        jsonfile: str = "./data/train_split.json",
        libraryfile: str = "./data/train_otsu_512_100_map.tar",
        transform: Any = None,
        size: int = 512,
        augment: bool = True,
        M: int = 2,
        ):

            super(IntraInstanceCausalDataset, self).__init__()
            assert M in [1, 2], "M should be either 1: 'weak' or 2: 'mild'"
            if transform is None and augment:
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.125, 0.125),
                            saturation=(0.875, 1.125),
                            brightness=(0.875, 1.2),
                            contrast=0.2,
                            p=0.5,
                        ),
                        HEDJitter(betas=(0.1, 0.0075), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.35),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.35),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
 
            elif transform is not None and augment:
                self.transform = transform
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ])

            with open(jsonfile) as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.slideIDX = data["slideIDX"]
                self.targets = data["targets"]
           
            lib = torch.load(libraryfile)
            
            print('Number of tiles: {}'.format(len(self.grid)))
            self.slidenames = lib['slides']
            self.mode = 1
            self.mult = lib['mult']
            self.targets2 =lib['slide_target']
            self.size = int(np.round(size*lib['mult']))
            self.level = lib['level']
            
            self.slideIDX = np.array(self.slideIDX)
            self.grid = np.array(self.grid)
            
            self.targets = np.array(self.targets)
            print(lib['level'])
            self.level = lib['level']
            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.0),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ])

            self.random_scale_rotate = A.Compose([
                A.Rotate((-90, 90), p=0.5),
                A.RandomScale(scale_limit=(-0.2, 1.2), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ])
            self.M = M
            self.augment = augment

    def __getitem__(self,index):
        
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]

        slide_ids = list(np.where(np.array(self.slideIDX)==slideIDX)[0])
        # ref_coords: coord of all neibhouring tiles (8 connectivity)
        ref_coords = [
            (coord[0] - self.size, coord[1] - self.size),
            (coord[0] - self.size, coord[1]),
            (coord[0] - self.size, coord[1] + self.size),
            (coord[0], coord[1] - self.size),
            (coord[0], coord[1] + self.size),
            (coord[0] + self.size, coord[1] - self.size),
            (coord[0] + self.size, coord[1]),
            (coord[0] + self.size, coord[1] + self.size)]
        
        # remove from ref_coords regions where there is a lot of whitish background
        ctx_coords = [i for i in slide_ids if (self.grid[i][0], self.grid[i][1]) in ref_coords]
        
        # ensure only tiles completely filled with tisue are used
        # if no neighbour is eligible (e.g., due to Otsu thresholding in S in HSV colour space)
        # return the tile itself
       
        if len(ctx_coords) == 0:
            j = index
        else:
            j = random.choice(ctx_coords)
        ctx_coord = self.grid[j]
        
        try:
            slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
        except:
            slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500"))
        
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img = np.array(img)
            if self.M == 1:
                img = self.rotate(image=img)["image"]
            else:
                img = self.random_scale_rotate(image=img)["image"]
        else:
            img = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img = np.array(img)
            
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img2 = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img2 = np.array(img2)
            if self.M == 1:
                img2 = self.rotate(image=img2)["image"]
            else:
                img2 = self.random_scale_rotate(image=img2)["image"]
        else:
            img2 = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img2 = np.array(img2)
        
        if (random.random() < 0.5 and self.augment):
            ctx_coord = [ctx_coord[0]-256, ctx_coord[1]-256]
            ctx = slide.read_region(ctx_coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            ctx = np.array(ctx)
            if self.M == 1:
                ctx = self.rotate(image=ctx)["image"]
            else:
                ctx = self.random_scale_rotate(image=ctx)["image"] 
        else:
            ctx = slide.read_region(ctx_coord, self.level,(self.size, self.size)).convert('RGB')
            ctx = np.array(ctx)
      
        if index < len(self.grid)-1:
            next_slideIDX = self.slideIDX[index+1]
            if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                slide.close()
                del slide
        else:
            slide.close()
            del slide
        if self.mult != 1:
            img = img.resize((224, 224),Image.BILINEAR)
            ctx = ctx.resize((224, 224),Image.BILINEAR)

        img = self.transform(image=img)["image"]
        ctx = self.transform(image=ctx)["image"]
        img2 = self.transform(image=img2)["image"]
  
        return img, img2, ctx, self.targets[index], self.targets[j]
    
    def __len__(self):
        return len(self.grid)
    
    
class MSCausalDataset(Dataset):

    def __init__(
        self,
        jsonfile: str = "./data/train_split.json",
        libraryfile: str = "./data/train_otsu_512_100_map.tar",
        transform: Any = None,
        size: int = 512,
        augment: bool = True,
        M: int = 2,
        train: bool=True,
        imp: bool=True,
        causal: bool=False
        ):

            super(MSCausalDataset, self).__init__()
            assert M in [1, 2], "M should be either 1: 'weak' or 2: 'mild'"
            if transform is None and augment:
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ],
                    additional_targets={"image0": "image"})
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.15, 0.15),
                            saturation=(0.825, 1.15),
                            brightness=(0.85, 1.2),
                            contrast=0.3,
                            p=0.5,
                        ),
                        HEDJitter(betas=(0.125, 0.01), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ],
                    additional_targets={"image0": "image"})
 
            elif transform is not None and augment:
                self.transform = transform
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"})

            with open(jsonfile) as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.slideIDX = data["slideIDX"]
                self.targets = data["targets"]
           
            lib = torch.load(libraryfile)
            
            print('Number of tiles: {}'.format(len(self.grid)))
            self.slidenames = lib['slides']
            self.mode = 1
            self.mult = lib['mult']
            self.targets2 =lib['slide_target']
            self.size = int(np.round(size*lib['mult']))
            self.level = lib['level']
            
            self.slideIDX = np.array(self.slideIDX)
            self.grid = np.array(self.grid)
            
            self.targets = np.array(self.targets)
            print(lib['level'])
            self.level = lib['level']
            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.0),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ],
                additional_targets={"image0": "image"})

            self.random_scale_rotate = A.Compose([
                A.Rotate((-180, 180), p=0.5),
                A.RandomScale(scale_limit=(-0.3, 1.3), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ],
                additional_targets={"image0": "image"})
            self.resize = A.Resize(size, size)
            self.M = M
            self.augment = augment
            self.imp = imp
            self.train = train
            self.causal = causal

    def __getitem__(self,index):
        
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]

        coord_j = (coord[0] - self.size, coord[1] - self.size)

        if self.imp:
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
            except:
                if self.train:
                    slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500"))
                else:
                    slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/non_annotated", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500"))
        else:
            slide = open_slide(self.slidenames[slideIDX])
        
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img = np.array(img)
            
            coord_j = [coord_j[0]-256, coord_j[1]-256]
            img_j = slide.read_region(coord_j, self.level, (self.size*4, self.size*4)).convert('RGB')
            img_j = np.array(img_j)
            
            if self.M == 1:
                aug = self.rotate(image=img, image0=img_j)
                img = aug["image"]
                img_j = aug["image0"]
            else:
                aug = self.random_scale_rotate(image=img, image0=img_j)
                img = aug["image"]
                img_j = aug["image0"]
        else:
            img = slide.read_region(coord, self.level, (self.size, self.size)).convert('RGB')
            img = np.array(img)
            
            img_j = slide.read_region(coord_j, self.level, (self.size*3, self.size*3)).convert('RGB')
            img_j = np.array(img_j)
            
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img2 = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img2 = np.array(img2)
            
            coord_j = [coord_j[0]-256, coord_j[1]-256]
            img2_j = slide.read_region(coord_j, self.level, (self.size*4, self.size*4)).convert('RGB')
            img2_j = np.array(img2_j)
            
            if self.M == 1:
                aug =  self.rotate(image=img2, image0=img2_j)
                img2 = aug["image"]
                img2_j = aug["image0"]
            else:
                aug = self.random_scale_rotate(image=img2, image0=img2_j)
                img2 = aug["image"]
                img2_j = aug["image0"]
        else:
            img2 = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img2 = np.array(img2)
            
            img2_j = slide.read_region(coord_j, self.level, (self.size*3, self.size*3)).convert('RGB')
            img2_j = np.array(img2_j)
            

        img_j = self.resize(image=img_j)["image"]
        img2_j = self.resize(image=img2_j)["image"]
      
        if index < len(self.grid)-1:
            next_slideIDX = self.slideIDX[index+1]
            if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                slide.close()
                del slide
        else:
            slide.close()
            del slide

        aug = self.transform(image=img, image0=img_j)
        img = aug["image"]
        img_j = aug["image0"]
        
        if self.train and self.causal:
            aug = self.transform(image=img2, image0=img2_j)
            img2 = aug["image"]
            img2_j = aug["image0"]
            return img, img2, img_j, img2_j, self.targets[index]
        elif self.train:
            return img, img_j, self.targets[index]
        else:
            return img, img_j, self.targets[index], self.slidenames[slideIDX]
    
    def __len__(self):
        return len(self.grid)
    

class MultiViewInfoMaxDataset(Dataset):

    def __init__(
        self,
        jsonfile: str = "./data/train_split.json",
        libraryfile: str = "./data/train_otsu_512_100_map.tar",
        transform: Any = None,
        size: int = 512,
        augment: bool = True,
        imp: bool = False,
        M: int = 2,
        ):

            super(MultiViewInfoMaxDataset, self).__init__()
            assert M in [1, 2], "M should be either 1: 'weak' or 2: 'mild'"
            if transform is None and augment:
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.125, 0.125),
                            saturation=(0.875, 1.125),
                            brightness=(0.875, 1.2),
                            contrast=0.2,
                            p=0.5,
                        ),
                        HEDJitter(betas=(0.1, 0.0075), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.35),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.35),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
 
            elif transform is not None and augment:
                self.transform = transform
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ])

            with open(jsonfile) as f:
                data = json.load(f)
                self.grid = data["grid"]
                self.slideIDX = data["slideIDX"]
                self.targets = data["targets"]
           
            lib = torch.load(libraryfile)
            
            print('Number of tiles: {}'.format(len(self.grid)))
            self.slidenames = lib['slides']
            self.mode = 1
            self.mult = lib['mult']
            self.targets2 =lib['slide_target']
            self.size = int(np.round(size*lib['mult']))
            self.level = lib['level']
            
            self.slideIDX = np.array(self.slideIDX)
            self.grid = np.array(self.grid)
            
            self.targets = np.array(self.targets)
            print(lib['level'])
            self.level = lib['level']
            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.0),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ])

            self.random_scale_rotate = A.Compose([
                A.Rotate((-90, 90), p=0.5),
                A.RandomScale(scale_limit=(-0.2, 1.2), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ])
            self.M = M
            self.augment = augment
            self.imp = imp

    def __getitem__(self,index):
        
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]
        if self.imp:
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
            except:
                slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500"))
        else:
            slide = open_slide(self.slidenames[slideIDX])
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img = np.array(img)
            if self.M == 1:
                img = self.rotate(image=img)["image"]
            else:
                img = self.random_scale_rotate(image=img)["image"]
        else:
            img = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img = np.array(img)
            
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img2 = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img2 = np.array(img2)
            if self.M == 1:
                img2 = self.rotate(image=img2)["image"]
            else:
                img2 = self.random_scale_rotate(image=img2)["image"]
        else:
            img2 = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            img2 = np.array(img2)
        
        if index < len(self.grid)-1:
            next_slideIDX = self.slideIDX[index+1]
            if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                slide.close()
                del slide
        else:
            slide.close()
            del slide
        if self.mult != 1:
            img = img.resize((224, 224),Image.BILINEAR)
            img2 = img2.resize(224, 224, Image.BILINEAR)

        img = self.transform(image=img)["image"]
        img2 = self.transform(image=img2)["image"]
  
        return img, img2

    def __len__(self):
        return len(self.grid)


class HEDJitter(ImageOnlyTransform):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangles the haematoxylin and eosin color channels by color deconvolution using a fixed matrix.
    Second, it perturbes the haematoxylin, eosin and DAB stains independently.
    Third, it transforms the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         betti is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \betti**
    """
    def __init__(self,
                 betas=(0.005, 0.005),
                 always_apply=False,
                 p=0.5):
        super(ImageOnlyTransform, self).__init__(always_apply, p)
        self.betas = betas


    def apply(self, image, **params):
        return self.adjust_hed(image)

    def adjust_hed(self, img):
        
        s = np.reshape(color.rgb2hed(img), (-1, 3))

        alpha = np.random.uniform(1 - self.betas[0], 1 + self.betas[0], (1, 2))
        betti = np.random.uniform(-self.betas[1], self.betas[1], (1, 2))

        # modify less the last channel as it is a residual in H&E-stained slides
        # in the original method, this channel encodes diaminobenzidine(DAB) staining
        # strong augmentation in this channel is equivalent to adding an artificial
        # DAB stain
        alpha = np.expand_dims(np.append(alpha, np.random.uniform(0.995, 1.005)), 0)
        betti = np.expand_dims(np.append(betti, np.random.uniform(-0.005, 0.005)), 0)

        
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))
        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin + 1e-6)).astype('uint8')  # rescale to [0,255]
    
        return rsimg

    def apply(self, img, **params):
        return self.adjust_hed(img)
    
    def get_params_dependent_on_targets(self, params):
        return {"betas": self.betas}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "betas"

class MILDataset(Dataset):

    def __init__(
        self,
        libraryfile: str = "./data/test_otsu_512_100_10k.tar",
        is_vit=False,
        augment=False,
        transform: Any = None,
        size: int = 512,
        M: int = 0,
        imp: bool = False,
        return_index: bool = False,
        train: bool = False,
        norm: bool = False,
        ):
            
            super(MILDataset, self).__init__()

            if transform is None and augment:
                
                if M == 1:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        HEDJitter(p=0.5),
                        A.Compose([
                            A.RandomScale(scale_limit=(-0.04, -0.01), p=0.5), # output size is different from input...
                            A.Resize(size, size)]),  # resize to original size
                        A.HueSaturationValue(
                            hue_shift_limit=5,
                            sat_shift_limit=5,
                            val_shift_limit=5, p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                else:
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ColorJitter(
                            hue=(-0.1, 0.1),
                            saturation=(0.875, 1.125),
                            brightness=(0.875, 1.2),
                            contrast=0.2,
                        ),
                        HEDJitter(betas=(0.15, 0.01), p=0.5),
                        A.GaussianBlur((15, 15), (0.1, 2.0), p=0.5),
                        A.GaussNoise(var_limit=(0 * 255, 0.1 * 255), p=0.5),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                        ToTensorV2(),
                    ])
                
            elif transform is not None:
                self.transform = transform
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.1, 0.1, 0.1)),
                    ToTensorV2(),
                ])

            self.augment = augment
            lib = torch.load(libraryfile)
            self.train = train
            self.slidenames = lib['slides']
            self.mode = 1
            self.mult = lib['mult']
            self.level = lib['level']
            self.slides = lib["slides"]
            
            # Flatten grid
            grid = []
            slideIDX = []
            for i, g in enumerate(lib['grid']):
                grid.extend(g)
                slideIDX.extend([i]*len(g))
            grid = np.array(grid)
            slideIDX = np.array(slideIDX)
            print('Number of tiles: {}'.format(len(grid)))

            self.targets = np.array(lib['slide_target'])

            self.slideIDX = slideIDX
            self.grid = np.array(grid)
            self.size = size

            self.rotate = A.Compose([
                A.Rotate((-30, 30), always_apply=True, p=1.),
                A.CenterCrop(self.size, self.size, always_apply=True, p=1.0),
                ])

            self.random_scale_rotate = A.Compose([
                A.Rotate((-90, 90), p=0.5),
                A.RandomScale(scale_limit=(-0.15, 1.15), p=0.5),
                A.Resize(2*size, 2*size),  # resize to original size
                A.CenterCrop(size, size, always_apply=True, p=1.0),
                ])
            self.M = M
            self.imp = imp
            self.return_index = return_index
            if norm:
                
                norm = VahadaneNormalizer() 
                
                f = os.path.join(
                        "./data/stain_norm_targets",
                        "image1.png"
                )
                target = Image.open(f)
                target = np.array(target)
                norm.fit(target)
            else:
                norm = None
            self.norm = norm
            self.is_vit = is_vit
            self.image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")


    def __getitem__(self, index):
        
        slideIDX = self.slideIDX[index]
        coord = self.grid[index]
        if self.imp:
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
            except:
                slide = open_slide(self.slidenames[slideIDX].replace("/home/imp-data/uploads/non_annotated", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500"))
        else:
            slide = open_slide(self.slidenames[slideIDX])
        if (random.random() < 0.5 and self.augment):
            coord = [coord[0]-256, coord[1]-256]
            img = slide.read_region(coord, self.level, (self.size*2, self.size*2)).convert('RGB')
            img = np.array(img)
            if self.M == 1:
                img = self.rotate(image=img)["image"]
            else:
                img = self.random_scale_rotate(image=img)["image"]
        else:
            img = slide.read_region(coord, self.level,(self.size, self.size)).convert('RGB')
            if not self.is_vit:
                img = np.array(img)
        
        if self.norm is not None:
            img = self.norm.transform(img)
        
        if index < len(self.grid)-1:
            next_slideIDX = self.slideIDX[index+1]
            if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                slide.close()
                del slide
        else:
            slide.close()
            del slide
        if self.mult != 1:
            img = img.resize((224, 224),Image.BILINEAR)

        if self.transform is not None and not self.is_vit:
            img = self.transform(image=img)["image"]
        if self.return_index:
            return img, self.targets[slideIDX], self.slidenames[slideIDX], index
        if not self.train:
            if self.is_vit:
                inputs = self.image_processor(img, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].squeeze()
                return inputs, self.targets[slideIDX], self.slidenames[slideIDX]
            return img, self.targets[slideIDX], self.slidenames[slideIDX]
        else:
            return img, self.targets[slideIDX]
    
    def __len__(self):
        return len(self.grid)


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform=None, size=224):
        lib = torch.load(libraryfile)
        # slides = []
        for i, name in enumerate(lib['slides']):
            #slides.append(openslide.OpenSlide(name))

            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
        print('')

        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))
        grid = np.array(grid)
        slideIDX = np.array(slideIDX)
        print('Number of tiles: {}'.format(len(grid)))

        self.slidenames = lib['slides']
        # self.slides = slides
        self.targets = np.array(lib['targets'])
        print(len(self.targets))
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.targets2 =np.array(lib['slide_target']) 
        self.size = int(np.round(size*lib['mult']))
        self.level = lib['level']
        self.selected_tiles = {}
        self.mapping_ids = []
        for slideId in self.slideIDX:
            if not(slideId in self.selected_tiles.keys()):
                self.selected_tiles[slideId] = [1]
            self.selected_tiles[slideId].append(0)
            self.selected_tiles[slideId][0] += 1
            self.mapping_ids.append(int(len(self.selected_tiles[slideId]) - 2))
        self.tiles_not_used_before = []
        self.different_tiles_used = []

        with open('tile_data.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["N_Slides","N_Tiles", "N_Tiles_Used","New_Tiles"])
        
        with open('missing_tiles_100.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Missing","Epoch", "Epoch_Missing"])

        with open('missing_tiles_50.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Missing","Epoch", "Epoch_Missing"])
        
        with open('missing_tiles_200.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Missing","Epoch", "Epoch_Missing"])
        
        with open('missing_tiles_500.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Missing","Epoch", "Epoch_Missing"])
        
        with open('missing_tiles_75.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(["Missing","Epoch", "Epoch_Missing"])

    def setmode(self,mode):
        self.mode = mode

    def save_tile_info(self): 
        with open('tile_data.csv', 'a') as f:
            write = csv.writer(f)
            #write.writerow(["N_Slides","N_Tiles", "N_Tiles_Used","New_Tiles"])
            write.writerow([len(self.selected_tiles.keys()),
                                len(self.slideIDX),
                                self.different_tiles_used[len(self.different_tiles_used)-1],
                                self.tiles_not_used_before[len(self.tiles_not_used_before)-1]])

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[x],self.targets2[self.slideIDX[x]]) for x in idxs]
        tiles_not_used_before = 0
        """
        for x in idxs:
            id_ = self.slideIDX[x]
            if self.selected_tiles[id_][self.mapping_ids[x]] == 0:
                tiles_not_used_before += 1
            self.selected_tiles[id_][self.mapping_ids[x]] += 1
             
        self.tiles_not_used_before.append(tiles_not_used_before)
        if len(self.different_tiles_used) == 0: 
            self.different_tiles_used.append(tiles_not_used_before)
        else:
            self.different_tiles_used.append(int(self.different_tiles_used[len(self.different_tiles_used)-1] + tiles_not_used_before))
        """
        #self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[x]) for x in idxs]

    def define_sampled_data(self,topk):
        self.slideIDX = self.slideIDX[topk]
        self.grid = self.grid[topk]
        self.targets = self.targets[topk]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))

    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/"))
            except:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/").replace("non_annotated","extra_500"))
            
            img = slide.read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if index < len(self.grid)-1:
                next_slideIDX = self.slideIDX[index+1]
                if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                    slide.close()
                    del slide
            
            else:
                slide.close()
                del slide
            
            if self.mult != 1:
                img = img.resize((224,224), Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(image=np.array(img))["image"]
            return img
        
        elif self.mode == 2:
            slideIDX, coord,target, target2 = self.t_data[index]
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/"))
            except:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/").replace("non_annotated","extra_500"))
            
            img = slide.read_region(coord,self.level,(self.size,self.size)).convert('RGB')

            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(image=np.array(img))["image"]

            if index < len(self.t_data):
                next_slideIDX = self.slideIDX[index+1]
                if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                    slide.close()
                    del slide
            
            else:
                slide.close()
                del slide
            if target == -1:
                return img, target2
            else:
                return img,target2
    
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)

        elif self.mode == 2:
            return len(self.t_data)


class MILdataset_pretraining(data.Dataset):
    def __init__(self, libraryfile='', transform=None, size = 512):
        lib = torch.load(libraryfile)
        # slides = []
        for i, name in enumerate(lib['slides']):
            #slides.append(openslide.OpenSlide(name))

            sys.stdout.write('Opening SVS headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
        print('')

        #Flatten grid
        grid = []
        slideIDX = []
        for i,g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g))
        
        print('Number of tiles: {}'.format(len(grid)))

        self.slidenames = lib['slides']
        # self.slides = slides
        self.targets = lib['targets']
        print(len(self.targets))
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = 1
        self.mult = lib['mult']
        self.targets2 = lib['slide_target']
        self.size = int(np.round(size*lib['mult']))
        self.level = lib['level']

        to_remove = []

        for i,target in enumerate(self.targets):
            if target == -1:
                to_remove.append(i)
        to_remove_targets2 = []
        for index in sorted(to_remove, reverse=True):
            del self.grid[index]
            del self.targets[index]
            to_remove_targets2.append(self.slideIDX[index])
            del self.slideIDX[index]
            
        self.slideIDX = np.array(self.slideIDX)
        self.targets2 = np.delete(self.targets2,list(set(to_remove_targets2)))
        print(self.targets2.shape)
        self.grid = np.array(self.grid)
        
        self.targets = np.array(self.targets)
        self.targets2 = np.array(self.targets2)

    def setmode(self,mode):
        self.mode = mode



    def __getitem__(self,index):
        if self.mode == 1:
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            try:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/"))
            except:
                slide = open_slide(self.slidenames[slideIDX].replace("uploads/","uploads/CRC/").replace("non_annotated","extra_500"))
            img = slide.read_region(coord,self.level,(self.size,self.size)).convert('RGB')

            if index < len(self.grid)-1:
                next_slideIDX = self.slideIDX[index+1]
                if self.slidenames[next_slideIDX] != self.slidenames[slideIDX]:
                    slide.close()
                    del slide
            
            else:
                slide.close()
                del slide
            
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)

            if self.transform is not None:
                img = self.transform(image=np.array(img))["image"]
            return img,self.targets[index]
    
    def __len__(self):
        if self.mode == 1:
            return len(self.grid)
       

class WSIEmbeddingsDataset(data.Dataset):
    def __init__(self, path, jsonfile, libraryfile):
        self.path = path
        with open(jsonfile, "r") as f:
          self.data = json.load(f)
        lib = torch.load(libraryfile)
        self.targets= {
            slide.split(".")[0].split("/")[-1]: lib["slide_target"][i]
            for i, slide in enumerate(lib["slides"])
        }


    def __getitem__(self, index):
        
        f = os.path.join(self.path, self.data[index])
        slide = self.data[index].split(".")[0]
        h = np.array(
            pd.read_csv(f)
        )
        h = torch.tensor(h).to(torch.float32)
        #TODO: delete this line
        h = h[..., :320]
        target = torch.tensor(self.targets[slide])
        
        return h, target.to(torch.int64)
        
    def __len__(self):
        return len(self.data)

