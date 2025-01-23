import datetime
import os
from typing import Any

import numpy as np
import torch

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from tqdm import tqdm

import argparse

# import wandb
# from data.datasets import WSIEmbeddingsDataset
import torch.utils.data as data

from models.transmil import TransMIL
from optimizers.lookahead import Lookahead
from utils.seeds import seed_worker
from utils.qwk import *
import json
import torch

import torch.nn as nn
import pandas as pd
import numpy as np

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class WSIEmbeddingsDataset(data.Dataset):
    def __init__(self, path, jsonfile, libraryfile, aug=False):
        self.path = path
        with open(jsonfile, "r") as f:
          self.data = json.load(f)
        lib = torch.load(libraryfile)
        self.targets = {
            slide.split(".")[0].split("/")[-1]: lib["slide_target"][i]
            for i, slide in enumerate(lib["slides"])
        }
        self.aug = aug

    def __getitem__(self, index):

        f = os.path.join(self.path, self.data[index])

        if self.aug:
            slide = ("-").join(self.data[index].split(".")[0].split("-")[:-1])
        else:
            slide = self.data[index].split(".")[0]
        h = np.array(
            pd.read_csv(f)
        )
        h = torch.tensor(h).to(torch.float32)
        
        target = torch.tensor(self.targets[slide])

        return h, target.to(torch.int64)

    def __len__(self):
        return len(self.data)


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--train_lib", type=str, default="<path-to-lib>")
    parser.add_argument("--val_lib", type=str, default="<path-to-lib>")
    parser.add_argument("--train_lib2", type=str, default="./data/BernCRC_test_otsu_512_100_map3.tar")
    parser.add_argument("--val_lib2", type=str, default="./data/BernCRC_test_otsu_512_100_map3.tar")
    parser.add_argument(
        "--train_path",
        type=str,
        default="/nas-ctm01/datasets/public/CRC_DG/11-15-2024-15:57:02-resnet34-567/crs10k-norm05-gan-bern/"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="/nas-ctm01/datasets/public/CRC_DG/11-15-2024-15:57:02-resnet34-567/crs10k-norm05-gan-bern/"
    )
    parser.add_argument(
        "--train_path2",
        type=str,
        default="/nas-ctm01/datasets/public/CRC_DG/fixmatch-005-01-15-2024-10:53:31-resnet34-567/bern-crc-fixmatch-005/"
    )
    parser.add_argument(
        "--val_path2",
        type=str,
        default="/nas-ctm01/datasets/public/CRC_DG/fixmatch-005-01-15-2024-10:53:31-resnet34-567/bern-crc-fixmatch-005/"
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/train_fold_1_otsu_512_100_1k.json",
    )
    parser.add_argument(
        "--valid_json",
        type=str,
        default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/valid_otsu_512_100.json",
    )
    parser.add_argument(
        "--train_json2",
        type=str,
        default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_train_split_fold_5.json",
    )
    parser.add_argument(
        "--valid_json2",
        type=str,
        default="/nas-ctm01/homes/jdfernandes/cadpath-ssl/data/train_test_splits/bern_val_split_fold_5.json",
    )
    parser.add_argument("--fc_size", type=int, default=512) # phikon: 768 plip: 512 gigapath: 1536 uni: 
    parser.add_argument(
        "--model",
        type=str,
        default="./saved/TransMIL/10-13-2023-13:33:16_transMIL-resnet34_single_best.pth")
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--trial", type=str, default="train-val-BERN")
    parser.add_argument("--lmbd", type=float, default=0.005)
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multi_source", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lmbd_", type=float, default=0.3)
    parser.set_defaults(multi_source=False, finetune=False)


    return parser.parse_args()

def train(
    model: Any,
    trainloader,
    valloader,
    trainloader2=None,
    valloader2=None,
    lmbd: float = 0.005,
    save_dir: str = "./saved/TransMIL/",
    output: str = "transMIL-resnet34",
    epochs: int = 50,
    accum_iter: int = 30,
    device: Any = torch.device("cuda"),
    save_every: int = 30,
    num_classes: int = 3,
    lr: float = 1e-5,
    wdecay: float = 1e-5,
    trial: str = "train-test-hun-crc",
    lmbd_: float = 0.5,
    ):
    
    config = {
    "base_learning rate": lr,
    "base_optimizer": "RAdam",
    "optimizer": "Lookahead",
    "epochs": 50,
    "batch_size": 1,
    "optim": "lookahead",
    "weight decay": wdecay,
    "lmbd": lmbd_,
    "loss": "QWK",
    "scheduler": "reduce_lr_on_plateau",
    "backbone": "resnet34",
    }
 
    TIMESTAMP = datetime.datetime.now()
    TIMESTAMP = TIMESTAMP.strftime("%m-%d-%Y-%H:%M:%S")
    
    basedir = os.path.join(
        save_dir,
        trial,
        TIMESTAMP
    )
    os.makedirs(basedir)
    
    with open(os.path.join(basedir, "config.json"), "w") as f:
        json.dump(config, f)
    
    base_optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=lr,
        weight_decay=wdecay,
    )
    optimizer = Lookahead(base_optimizer, alpha=0.5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=5e-3,
        patience=20,
        mode="max"
    )
    scaler = GradScaler()

    print("Begin train...")
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1} of {epochs}...")
        ce_losses = []
        
        if trainloader2 is not None:
            tr2_iterator = iter(trainloader2)
        for step, (h, target) in enumerate(tqdm(trainloader)):
            if trainloader2 is not None:
                try:
                    data2 = next(tr2_iterator)
                except StopIteration:
                    tr2_iterator = iter(trainloader2)
                    data2 = next(tr2_iterator)
                h2, target2 = data2
                h2 = h2.to(device)
                target2 = target2.to(device)
            
            h = h.to(device)
            target = target.to(device)
            
            with torch.cuda.amp.autocast():
                
                optimizer.zero_grad()

                out = model(h)
                
                if trainloader2 is not None:
                    out2 = model(h2)
                grads = []

                if trainloader2 is not None:
                    loss = lmbd_*torch.sum(
                        torch.stack([
                            criterion(out["logits"][i], target) 
                            for i in range(len(out["logits"]))
                                    ])) + (1-lmbd_)*torch.sum(
                        torch.stack([
                            criterion(out2["logits"][i], target2) 
                            for i in range(len(out2["logits"]))
                                    ]))
  
                else:
                    loss = torch.sum(
                            torch.stack([
                                criterion(out["logits"][i], target)
                                for i in range(len(out["logits"]))
                                    ]))

            scaler.scale(loss).backward(retain_graph=True)
            for module in model.cls_heads:
                module_grads = torch.tensor([], device=device)
                for param in module.parameters():
                    module_grads = torch.cat((module_grads, param.grad.view(-1)))
                grads.append(module_grads)
            grads = torch.stack(grads)
            
            
            dot = torch.inner(grads, grads)
            dot[torch.eye(dot.shape[0], dot.shape[1]).to(torch.bool)] = 0

            loss_diversity = loss + lmbd*torch.sum(dot)/2
            # remove cross entropy loss grads
            optimizer.zero_grad()  
            # replace standard cross entropy loss with diversity regularization         
            scaler.scale(loss_diversity).backward()
            scaler.step(optimizer)

            scaler.update()

            ce_losses.append(loss.item())

            if ((step +1) % accum_iter == 0) or ((step + 1) == len(trainloader)):
                
                ce_l = np.mean(ce_losses)


            if (step + 1) % save_every == 0 or ((step + 1) == len(trainloader)):
                # save model every 5000 iters
                f = os.path.join(basedir, TIMESTAMP+'_'+output+".pth")
                torch.save(model.state_dict(), f)

        pred = []
        true = []
        model.eval()
        preds_= []
        for step, (h, target) in enumerate(tqdm(valloader)):
            
            h = h.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast():
                with torch.no_grad():   
                    out = model(h)
                                    
                    probas = torch.mean(torch.stack(out["probas"]), 0)
                    pred.append(torch.argmax(probas, dim=1))
                    preds_.append(torch.tensor([torch.argmax(prob, dim=1) for prob in out["probas"]]))
                    true.append(target)



        if valloader2 is not None: 
            pred2 = []
            true2 = []
            preds2_= [] 

            for step, (h2, target2) in enumerate(tqdm(valloader2)):
                h2 = h2.to(device)
                target2 = target2.to(device)

                with torch.cuda.amp.autocast():
                    with torch.no_grad():   
                        out = model(h2)
                                        
                        probas = torch.mean(torch.stack(out["probas"]), 0)
                        pred2.append(torch.argmax(probas, dim=1))
                        preds2_.append(torch.tensor([torch.argmax(prob, dim=1) for prob in out["probas"]]))
                        true2.append(target2)
                    
        preds_= torch.stack(preds_)
        pred = torch.flatten(torch.stack(pred))
        true = torch.flatten(torch.stack(true))

        if valloader2 is not None:
            preds2_= torch.stack(preds2_)
            pred2 = torch.flatten(torch.stack(pred2))
            true2 = torch.flatten(torch.stack(true2))
            qwk_val2 = quadratic_weighted_kappa(pred2.cpu(), true2.cpu(), 0, num_classes-1)
            qwk_single2 = [quadratic_weighted_kappa(preds2_[:,i].T.cpu(), true2.cpu(), 0, num_classes-1) for i in range(preds2_.shape[-1])]
            qwk_val2 = np.mean(qwk_val2)
            qwk_single_val2 = np.max(qwk_single2)
        
        qwk_val = quadratic_weighted_kappa(pred.cpu(), true.cpu(), 0, num_classes-1)
        qwk_single = [quadratic_weighted_kappa(preds_[:,i].T.cpu(), true.cpu(), 0, num_classes-1) for i in range(preds_.shape[-1])]
        qwk_val = np.mean(qwk_val)
        qwk_single_val = np.max(qwk_single)
        if valloader2 is not None:
            qwk_val = 0.5*(qwk_val + qwk_val2)
            qwk_single_val = 0.5*(qwk_single_val + qwk_single_val2)
        
        print(qwk_val)

        
        scheduler.step(torch.tensor(qwk_val).to(device))

        if epoch == 0:
            best_qwk = qwk_val
            best_qwk_single = qwk_single_val
        
        if qwk_val >= best_qwk:
            best_qwk = qwk_val
            f = os.path.join(basedir, TIMESTAMP+'_'+output+"_mean_best.pth")
            torch.save(model.state_dict(), f)
        
        if qwk_val >= best_qwk:
            best_qwk = qwk_val
            f = os.path.join(basedir, TIMESTAMP+'_'+output+"_mean_best.pth")
            torch.save(model.state_dict(), f)
        if qwk_single_val >= best_qwk_single:
            best_qwk_single = qwk_single_val
            f = os.path.join(basedir, TIMESTAMP+'_'+output+"_single_best.pth")
            torch.save(model.state_dict(), f)


        m = { 
         "qwk (mean)": np.around(qwk_val, decimals=4, out=None).item(),
         "qwk single (mean)": np.around(qwk_single_val, decimals=4, out=None).item(),
         "qwk (best)": np.around(best_qwk, decimals=4, out=None).item(),
         "qwk single (best)": np.around(best_qwk_single, decimals=4, out=None).item(),
         "loss": ce_l 
        }
        
        with open(
            os.path.join(
                basedir,"train-val-metrics.json"), "w") as f:
            json.dump(m, f)

def main(
        num_classes: int = 3,
        batch_size: int = 1,
        workers: int = 8,
        save_dir: str = "./saved/TransMIL/",
        output: str = "transMIL-resnet34",
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}")
    args = parse_args()

    trainset = WSIEmbeddingsDataset(path=args.train_path, jsonfile=args.train_json, libraryfile=args.train_lib, aug=False)
    valset = WSIEmbeddingsDataset(path=args.val_path, jsonfile=args.valid_json, libraryfile=args.val_lib)

    if args.multi_source:
        trainset2 = WSIEmbeddingsDataset(path=args.train_path2, jsonfile=args.train_json2, libraryfile=args.train_lib2)
        valset2 = WSIEmbeddingsDataset(path=args.val_path2, jsonfile=args.valid_json2, libraryfile=args.val_lib2)
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
        generator=g,
        prefetch_factor=128,
    )

    valloader = DataLoader(
        valset,
        drop_last=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        prefetch_factor=128,
        )

    if args.multi_source:

        trainloader2 = DataLoader(
            trainset2,
            drop_last=True,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )

        valloader2 = DataLoader(
            valset2,
            drop_last=True,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
            )
    else:
        trainloader2 = None
        valloader2 = None

    
    if args.finetune:
        model = TransMIL(num_classes=num_classes, size=args.fc_size, n_heads=1)
        model = model.to(device)
        model.load_state_dict(
                torch.load(args.model, map_location=device)
        )
        # model.cls_heads = torch.nn.ModuleList([
        #     torch.nn.Linear(args.fc_size, num_classes) for _ in range(args.n_heads)]).to(device)
        model.train()
    else:
        model = TransMIL(num_classes=num_classes, size=args.fc_size, n_heads=args.n_heads)
        model = model.to(device)
    
    
    train(
        model,
        trainloader,
        valloader,
        trainloader2,
        valloader2,
        args.lmbd,
        save_dir,
        output,
        device=device,
        lr=args.lr,
        wdecay=args.wdecay,
        trial=args.trial,
        lmbd_=args.lmbd_,
)
    print("Execution finished!")

if __name__ == '__main__':
    main()
