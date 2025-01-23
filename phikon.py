import argparse
from collections import OrderedDict
import datetime
import os
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.models as models
import pandas as pd

from sklearn.metrics import confusion_matrix as confusion
from torch.utils.data import DataLoader


import data.datasets as myDatasets
from utils.seeds import seed_worker

import torch
from transformers import ViTModel
from PIL import Image

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def arg_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="TilesDataset")
    parser.add_argument("--jsonfile", type=str, default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_crc_test_otsu_512_100_map2_lib.json")
    parser.add_argument("--libraryfile", type=str, default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/BernCRC_test_otsu_512_100_map2.tar")
    parser.add_argument("--outdir", type=str, default="bern-crc-phikon")
    parser.add_argument("--save_every", type=int, default=10)
    parser.set_defaults()
    args = parser.parse_args()
    return args

def save_as_csv(embeddings, path):
    keys = list(embeddings.keys())
    for key in keys:
        fname = key+".csv"
        f = os.path.join(
            path, fname)
        z = embeddings[key]
        z_np = z.numpy()
        df = pd.DataFrame(z_np)
        df.to_csv(f, index=False, header=False)


def inference(
    dataloader,
    args,
    device: Any = torch.device("cuda"),
    num_classes: int = 3,
    root: str = "/nas-ctm01/datasets/public/CRC_DG/"
    ):


    # load phikon
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model = model.to(device)

    path = os.path.join(
            root, "phikon-pretrained", args.outdir)

    if not os.path.exists(path):
        os.makedirs(path)
    
    print("Begin inference...")

    model.eval()

    embeddings = {}
    for step, batch in enumerate(dataloader):
        if len(batch)==2:
            key, inputs = batch
            inputs = inputs.to(device)
    
            with torch.no_grad():
                    outputs = model(**inputs)
                    embs = outputs.last_hidden_state[:, 0, :]  # (1, 768) shape
        else:
            resol_1, resol_2, target, key = batch
            resol_1 = resol_1.to(device)
            resol_2 = resol_2.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():  
                    embs = model(resol_1, resol_2)


        for i, z in enumerate(embs):
            if not key[i] in list(embeddings.keys()):
                
                # save every k WSI embedding predictions to reduce memory requirements
                if len(list(embeddings.keys())) % args.save_every == 0:
                    
                    save_as_csv(embeddings, path)
                    embeddings = {}
                      
                embeddings[key[i]] = z.cpu().unsqueeze(0)
            else:
                embeddings[key[i]] = torch.cat((embeddings[key[i]], z.cpu().unsqueeze(0)))

    # save the last batch of embeddings
    save_as_csv(embeddings, path)



def main(
        num_classes: int = 3,
        batch_size: int = 256,
        workers: int = 16,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args = arg_parser()
    dataset = getattr(myDatasets, args.dataset)
    dataset = dataset(
        path="BernCRC/tiles/level-0/",
        augment=False,
        transform=None,
        cons_reg=False,
        return_key=True,
    )
   
    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        drop_last=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=8,
        )
    print("Device: ", device)

    inference(dataloader, args, device)
    print("Execution finished!")



if __name__ == '__main__':
    main()
