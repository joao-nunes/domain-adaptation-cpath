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

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def parse_args()
    parser = ArgumentParser()
    parser.add_argument("--train_libraryfile", type=str, default="./data/train_otsu_512_100_10k.tar")
    parser.add_argument("--train_libraryfile2", type=str, default="./data/BernCRC_test_otsu_512_100_map2.tar")
    parser.add_argument("--val_libraryfile", type=str, default="./data/train_otsu_512_100_10k.tar")
    parser.add_argument("--val_libraryfile2", default="./data/BernCRC_test_otsu_512_100_map2.tar")
    parser.add_argument("--train_json", type=str, default="./data/train_test_splits/train-567.json")
    parser.add_argument("--train_json2", type=str, default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_train_split_fold_1_lib.json")
    parser.add_argument("--valid_json", type=str, default="./data/train_test_splits/valid.json")
    parser.add_argument("--valid_json2", type=str, default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_val_split_fold_1_lib.json")
    parser.add_argument("--resume", action=BooleanOptionalAction)
    parser.add_argument("--ckpt", type=str, default="./saved/baseline/09-20-2023-12:49:29_resnet34.pth")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_features", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--save_dir", type=str, default="./saved/fixmatch/")
    parser.add_argument("--output", type=str, default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--thr", type=float, default=0.9)
    parser.add_argument("--lmbd", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.9)
    return parser.parse_args()


def train(
    model: Any,
    trainloader1,
    valloader1,
    trainloader2,
    valloader2,
    optimizer: str = "sgd",
    save_dir: str = "./saved/fixmatch/",
    output: str = "resnet34",
    resume: bool=False,
    ckpt: str= "",
    epochs: int = 30,
    accum_iter: int = 1000,
    scheduler: str = "reduce_on_plateau",
    lr: float = 1e-4,
    device: Any = torch.device("cuda"),
    save_every: int = 5000,
    num_classes: int = 3,
    thr: float = 0.9,
    lmbd: float = 0.3,
    tau: float = 0.9,
    eps: float = 1e-6,
    warmup_steps: int= 5000,
    ):

    os.environ["WANDB_API_KEY"] = "e89a73b7cff70cb22160ed0b4ea0c24faa5b32af"
    
    wandb.init(project="cadpath-ssl",
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
        
        print(f"Epoch {epoch+1} of {epochs}...")
        losses = []
        losses_u = []
        tr2_iterator = iter(trainloader2)
        for step, data1 in enumerate(trainloader1):
            try:
                data2 = next(tr2_iterator)
            except StopIteration:
                tr2_iterator = iter(trainloader2)
                data2 = next(tr2_iterator)
            image, target = data1
            img_w, img_s = data2
            
            image = image.to(device)
            target = target.to(device)
            
            img_w = img_w.to(device)
            img_s = img_s.to(device)
            
            model.eval()
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = model(img_w)
                    probas = torch.softmax(logits, -1)
                
            # keep confident pseudo labels and corresponding 'strong' image augmentation
            pseudo = probas[torch.sum(probas >= thr, -1), ...]
            img_s = img_s[torch.sum(probas >= thr, -1), ...]
                
            # sharpening (soft pseudo-labelling)
            pseudo = (pseudo ** (1 / tau)) / (torch.sum(pseudo ** (1 / tau), 1, keepdim=True) + eps)
            model.train()  
                
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                
                logits = model(image)
                logits_u = model(img_s)
                loss_u = nn.functional.cross_entropy(logits_u, pseudo)
                loss_s = qwk(logits, target, num_classes)
                # 5000 warmup iterations
                if epoch == 0 and step <= warmup_steps:
                    loss =  loss_s
                # Consistency regularization with confident pseudo-labels
                else: 
                    loss =  loss_s + lmbd * loss_u
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss_s.item())
            losses_u.append(loss_u.item())

            if ((step +1) % accum_iter == 0) or ((step + 1) == len(trainloader1)):
                qwk_l = np.mean(losses)
                ce_l = np.mean(losses_u)
                try:
                    wandb.log({"Quadratic Weighted Kappa Loss (train)": qwk_l, "Consistency Loss (CE) (train)": ce_l})
                except:
                    print("Unexpected error logging QWK Loss.... QWK (train): ", str(qwk_l),
                          "Consistency Loss (CE) (train): ", str(ce_l)
                          )
                losses = []
                losses_u = []

            if (step + 1) % save_every == 0 or ((step + 1) == len(trainloader1)):
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
        losses_u = []
        
        vl2_iterator = iter(valloader2)
        for step, data1 in enumerate(valloader1):
            try:
                data2 = next(vl2_iterator)
            except StopIteration:
                vl2_iterator = iter(valloader2)
                data2 = next(vl2_iterator)
            
            image, target = data1
            image = image.to(device)
            target = target.to(device)
            
            img_w, img_s = data2
            img_w = img_w.to(device)
            img_s = img_s.to(device)
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    logits = model(img_w)
                    probas = torch.softmax(logits, -1)
                
            # keep confident pseudo labels and corresponding 'strong' image augmentation
            pseudo = probas[torch.sum(probas >= thr, -1), ...]
            img_s = img_s[torch.sum(probas >= thr, -1), ...]
                
            # sharpening (soft pseudo-labelling)
            pseudo = (pseudo ** (1 / tau)) / (torch.sum(pseudo ** (1 / tau), 1, keepdim=True) + eps)
            
            with torch.cuda.amp.autocast():
                with torch.no_grad():   
                        logits = model(image)
                        logits_u = model(img_s)
                        probas = torch.nn.functional.softmax(logits.float(), dim=1).float()
                        loss_u = nn.functional.cross_entropy(logits_u, pseudo)
            losses_u.append(loss_u.item())
            pred.append(torch.argmax(probas, dim=1))
            true.append(target)
                        
        pred = torch.flatten(torch.stack(pred))
        true = torch.flatten(torch.stack(true))
        qwk_val = quadratic_weighted_kappa(pred.cpu(), true.cpu(), 0, num_classes-1)
        ce_l = np.mean(losses_u)
          
        if qwk_val > best_qwk:
            best_qwk = qwk_val
            f = os.path.join(save_dir, TIMESTAMP+'_'+output+"_best.pth")
            torch.save(model.state_dict(), f)
        try:
            wandb.log({"Quadratic Weighted Kappa (val)": qwk_val, "Consistency Loss (CE) (val)": ce_l})
        except:
            print(
                "Unexpected error logging QWK metric.... QWK (val): ", str(qwk_val),
                "Consistency Loss (CE) (val): ", ce_l
                )


def main(
        train_libraryfile: str = "./data/train_otsu_512_100_10k.tar",
        train_libraryfile2: str = "./data/BernCRC_test_otsu_512_100_map2.tar",
        val_libraryfile: str = "./data/train_otsu_512_100_10k.tar",
        val_libraryfile2: str = "./data/BernCRC_test_otsu_512_100_map2.tar",
        train_json: str = "./data/train_test_splits/train-567.json",
        train_json2: str = "/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_train_split_fold_1_lib.json",
        valid_json: str = "./data/train_test_splits/valid.json",
        valid_json2: str = "/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_val_split_fold_1_lib.json",
        resume: bool = False,
        ckpt: str = "./saved/baseline/09-20-2023-12:49:29_resnet34.pth",
        num_classes: int = 3,
        in_features: int = 512,
        batch_size: int = 256,
        workers: int = 16,
        optimizer: str = "sgd",
        save_dir: str = "./saved/fixmatch/bern-crc-ablation",
        output: str = "resnet34",
        epochs: int = 30,
        lr: float = 1e-4,
        thr: float = 0.9,
        lmbd: float = 0.3,
        tau: float = 0.9,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    trainset = BaseDataset(
        jsonfile=train_json,
        libraryfile=train_libraryfile,
        augment=True,
        transform=None
    )
    valset = BaseDataset(
        jsonfile=valid_json,
        libraryfile=val_libraryfile,
        augment=True,
        transform=None
        )
    
    trainset2 = BaseDataset(
        jsonfile=train_json2,
        libraryfile=train_libraryfile2,
        augment=True,
        transform=None,
        multiview=True,
        imp=False,
    )
    
    valset2 = BaseDataset(
        jsonfile=valid_json2,
        libraryfile=val_libraryfile2,
        augment=True,
        transform=None,
        multiview=True,
        imp=False
        )

    g = torch.Generator()
    g.manual_seed(SEED)

    trainloader1 = DataLoader(
        trainset,
        drop_last=True,
        batch_size=batch_size//8,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        )
    
    trainloader2 = DataLoader(
        trainset2,
        drop_last=True,
        batch_size=batch_size//16,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        )

    valloader1 = DataLoader(
        valset,
        drop_last=True,
        batch_size=batch_size//8,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=8
        )
    
    valloader2 = DataLoader(
        valset2,
        drop_last=True,
        batch_size=batch_size//16,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=8
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
        trainloader1,
        valloader1,
        trainloader2,
        valloader2,
        optimizer,
        save_dir,
        output,
        resume=resume,
        ckpt=ckpt,
        epochs=epochs,
        lr=lr,
        thr=thr,
        lmbd=lmbd,
        tau=tau
    )
    print("Execution finished!")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.train_libraryfile,
        args.train_libraryfile2,
        args.val_libraryfile,
        args.val_libraryfile2,
        args.train_json,
        args.train_json2,
        args.valid_json,
        args.valid_json2,
        args.resume,
        args.ckpt,
        args.num_classes,
        args.in_features,
        args.batch_size,
        args.workers,
        args.optimizer,
        args.save_dir,
        args.output,
        args.epochs,
        args.lr,
        args.thr,
        args.lmbd,
        args.tau)
