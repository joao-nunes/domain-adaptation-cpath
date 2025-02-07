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
from tqdm import tqdm

from cpathutils.datasets import TilesDataset
from utils.qwk import *
from utils.seeds import seed_worker
from cycle_gan import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from utils.identity import Identity

def arg_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./saved/baseline/11-15-2024-15:57:02_resnet34_best.pth")
    parser.add_argument("--timestamp", type=str, default="11-15-2024-15:57:02")
    parser.add_argument("--name", type=str, default="resnet34-567")
    parser.add_argument("--libraryfile", type=str, default="./data/valid_otsu_512_100_10k.tar")
    parser.add_argument("--outdir", type=str, default="crs10k-norm05-gan-bern")
    parser.add_argument("--mode", type=int, default=0, help="defines test mode. '0' for baseline or '1' for i2cirl")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--generator", action=argparse.BooleanOptionalAction)
    parser.add_argument("--generator_path", type=str, default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/saved/cycle-gan/Gen1-no-data-aug_.pth")
    parser.add_argument("--tiles_path", type=str, default="/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC_Bern/tiles/level-0/")
    parser.set_defaults(generator=True)
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
    model: Any,
    dataloader,
    args,
    generator: Any = None,
    device: Any = torch.device("cuda"),
    num_classes: int = 3,
    root: str = "/nas-ctm01/datasets/public/CRC_DG/",
    n_transforms: int = 1
    ):

    modelname = args.timestamp+"-"+args.name
    path = os.path.join(
            root, modelname, args.outdir)

    if not os.path.exists(path):
        os.makedirs(path)
    
    print("Begin inference...")

    model.eval()

    embeddings = {}
        
    for j in range(n_transforms):
        
        for step, batch in enumerate(tqdm(dataloader)):
            
            if len(batch)==3:
                image, target, key = batch
                target = target.to(device)
            else:
                key, image = batch
            image = image.to(device)
            if generator is not None:
                image = generator(image)

            with torch.cuda.amp.autocast():
                with torch.no_grad():  
                        embs = model(image)
            for i, z in enumerate(embs):
                if n_transforms != 1:
                    key[i] = key[i] + "-" + str(j)

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
        batch_size: int = 8,
        workers: int = 16,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    args = arg_parser()
    transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ])
    dataset = TilesDataset(
        path=args.tiles_path,
        transform=transform,
        augment=False,
        return_key=True,
        N=1
        )
   

    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        drop_last=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2,
        )
    print("Device: ", device)


    if args.mode == 0:
        model = models.resnet34()
    else:
        model = I2CIRLResNet(
        num_classes=num_classes)

    model.fc = Identity()
    model = model.to(device)
        
    if args.generator:
        generator = Generator(img_channels=3, num_residuals=15).to(device)
        generator.load_state_dict(
            torch.load(
               args.generator_path,
               map_location=device)["model_state_dict"]
        )
    else: 
        generator=None

    state_dict = torch.load(
        args.ckpt,
        map_location=torch.device(device)
    )
    state_dict = OrderedDict([((".").join(k.split(".")[1:]), v) if k.startswith("0.") else (k, v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=False)
    inference(model, dataloader, args, generator=generator, device=device)
    print("Execution finished!")



if __name__ == '__main__':
    main()
