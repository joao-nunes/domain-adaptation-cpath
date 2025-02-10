import datetime
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from torchvision.models import ResNet34_Weights
from tqdm import tqdm

import wandb
from data.datasets import BaseDataset
from utils.qwk import *
from utils.seeds import seed_worker
from utils.identity import Identity

from data import datasets
import sys

from argparse import ArgumentParser
import argparse

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_library_file",type=str, default="./data/train_otsu_512_100_10k.tar")
    parser.add_argument("val_library_file", type=str, default="./data/train_otsu_512_100_10k.tar")
    parser.add_argument("--train_json", type=str, default="./data/train_test_splits/train-567.json")
    parser.add_argument("--valid_json", type=str, default="./data/train_test_splits/valid.json")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ckpt", type=str, default="./saved/baseline/09-20-2023-12:49:29_resnet34.pth")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--_in_features", type=int, defaault=512)
    parser.add_argumrnt("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type="int", default="8")
    parser..add_argument("--optimizer", type=str, default="sgd")
    parser.add_argumenr("--save_dir", type=str, default="./saved/baseline/09-20-2023-12:49:29_resnet34.pth")
    parser.add_argument("--output", type=str, default="resnet34")
    parser.add_argument("--mydataset", type=str, default="BaseDataset")
    parser.add_argument("--lr", type = float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.set_defaults(resume=False)
    return parser.parse_args()


def train(
    model: Any,
    trainloader,
    valloader,
    optimizer: str = "sgd",
    save_dir: str = "./saved/ms/",
    output: str = "resnet34",
    resume: bool=False,
    ckpt: str= "",
    epochs: int = 30,
    accum_iter: int = 500,
    lr: float = 1e-4,
    device: Any = torch.device("cuda"),
    save_every: int = 5000,
    num_classes: int = 3,
    ):

    os.environ["WANDB_API_KEY"] = "e89a73b7cff70cb22160ed0b4ea0c24faa5b32af"
    
    wandb.init(project="ms",
               entity="jdnunes",
               config = {
    "learning rate": 1e-4,
    "epochs": 50,
    "batch_size": 32,
    "optim": "sgd",
    "weight decay": 3e-4,
    "loss": "QWK",
    "model": "resnet34",
    })

    TIMESTAMP = datetime.datetime.now()
    TIMESTAMP = TIMESTAMP.strftime("%m-%d-%Y-%H:%M:%S")

    wandb.run.name = TIMESTAMP
    wandb.run.save("wandb")
    
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=3e-4,
        )
    if resume:
        ckpt = torch.load(ckpt, map_location=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]) 
    
    
    scaler = GradScaler()
    # wandb.watch(model)
    print("Begin train...")
    best_qwk = 0
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1} of {epochs}...")
        losses = []
        for step, batch in enumerate(tqdm(trainloader)):    
            
            if len(batch)==3:
                resol_1, resol_2, target = batch
                resol_1 = resol_1.to(device)
                resol_2 = resol_2.to(device)
                target = target.to(device)
            else:
                image, target = batch
                image = image.to(device)
                target = target.to(device)
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                logits = model(image)
                loss = qwk(logits, target, num_classes)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            
            if ((step +1) % accum_iter == 0) or ((step + 1) == len(trainloader)):
                qwk_l = np.mean(losses)
                try:
                    wandb.log({"Quadratic Weighted Kappa Loss (train)": qwk_l})
                except:
                    print("Unexpected error logging QWK Loss.... QWK (train):", str(qwk_l))
                losses = []

            if (step + 1) % save_every == 0 or ((step + 1) == len(trainloader)):
                # save model every 5000 iters
                f = os.path.join(save_dir, TIMESTAMP+'_'+output+".pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                        }, f)
            
        model.eval()
        pred = []
        true = []
        for step, batch in enumerate(tqdm(valloader)):
            
            image, target = batch
            image = image.to(device)
            target = target.to(device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():   
                    logits = model(image)
                    probas = torch.nn.functional.softmax(logits.float(), dim=1).float()
            pred.append(torch.argmax(probas, dim=1))
            true.append(target)
                        
        pred = torch.flatten(torch.stack(pred))
        true = torch.flatten(torch.stack(true))
        qwk_val = quadratic_weighted_kappa(pred.cpu(), true.cpu(), 0, num_classes-1)
                    
        if qwk_val > best_qwk:
            best_qwk = qwk_val
            f = os.path.join(save_dir, TIMESTAMP+'_'+output+"_best.pth")
            torch.save(model.state_dict(), f)
        try:
            wandb.log({"Quadratic Weighted Kappa (val)": qwk_val})
        except:
            print("Unexpected error logging QWK Loss.... QWK (val):", str(qwk_val))

def main(
        train_libraryfile: str = "./data/train_otsu_512_100_10k.tar",
        val_libraryfile: str = "./data/train_otsu_512_100_10k.tar",
        train_json: str = "./data/train_test_splits/train-567.json",
        valid_json: str = "./data/train_test_splits/valid.json",
        resume: bool = False,
        ckpt: str = "./saved/baseline/09-20-2023-12:49:29_resnet34.pth",
        num_classes: int = 3,
        in_features: int = 512,
        batch_size: int = 256,
        workers: int = 8,
        optimizer: str = "sgd",
        save_dir: str = "./saved/baseline/",
        output: str = "resnet34",
        multiscale: bool = False,
        mydataset: str = "BaseDataset",
        lr: float = 1e-4,
        epochs: int = 30,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


    dataset = getattr(datasets, mydataset)
    trainset = dataset(
        jsonfile=train_json,
        libraryfile=train_libraryfile,
        augment=True,
        transform=None,
    )
    valset = dataset(
        jsonfile=valid_json,
        libraryfile=val_libraryfile,
        augment=True,
        transform=None,
        )
    
    g = torch.Generator()
    g.manual_seed(SEED)

    trainloader = DataLoader(
        trainset,
        drop_last=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
        )

    valloader = DataLoader(
        valset,
        drop_last=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
        )
    

    model = models.resnet34(
            weights=ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(in_features, num_classes)
        
    model = model.to(device)
    if resume:
        ckpt_ = torch.load(ckpt,
                          map_location=device)
        model.load_state_dict(ckpt_["model_state_dict"])
    
    train(
        model,
        trainloader,
        valloader,
        optimizer,
        save_dir,
        output,
        resume=resume,
        ckpt=ckpt,
        lr=lr,
        epochs=epochs,
    )
    print("Execution finished!")



if __name__ == '__main__':
    args = parse_args()
    main(args.train_library_file,
        args.val_library_file,
        args.train_json,
        args.valid_json,
        args.resume,
        args.ckpt,
        args.num_classes,
        args.in_features,
        args.batch_size,
        args.workers,
        args.optimizer,
        args.save_dir,
        args.output,
        args.multiscale,
        args.mydataset,
        args.lr,
        args.epochs
        )
