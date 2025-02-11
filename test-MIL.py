import datetime
import os
from typing import Any

import numpy as np
import torch

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from tqdm import tqdm

import argparse

import wandb
# from data.datasets import WSIEmbeddingsDataset

from models.transmil import TransMIL
from models.clam import CLAM_SB
from utils.seeds import seed_worker
from utils.qwk import *

from sklearn.metrics import confusion_matrix as confusion
from torchmetrics import AUROC, ROC
from torchmetrics.classification import ConfusionMatrix
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from utils.mil import *

import json
import pandas as pd
import numpy as np
import torch.utils.data as data

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import sys

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        default="./data/BernCRC_test_otsu_512_100_map3.tar")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="transMIL-resnet34_single_best.pth")

    parser.add_argument(
        "--path",
        type=str,
        default="/nas-ctm01/datasets/public/CRC_DG/11-15-2024-15:57:02-resnet34-567/crs10k-norm05-gan-bern/"
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="./data/train_test_splits/bern_test_fold_5.json",
    )
    parser.add_argument("--run_id", type=str, default="12-04-2024-10:13:28")
    parser.add_argument("--aggregation", type=str, default="TransMIL")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--trial", type=str, default="train-val-IMP")
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--fc_size", type=int, default=512)
    parser.add_argument("--ensemble", action=argparse.BooleanOptionalAction)
    parser.add_argument("--source", type=str, default="bern-005")
    parser.set_defaults(ensemble=False)
    return parser.parse_args()


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
        target = torch.tensor(self.targets[slide])
        
        return h, target.to(torch.int64)
        
    def __len__(self):
        return len(self.data)

def test(
    model: Any,
    dataloader,
    run_id: str = "mm-dd-yyyy-hh:mm:ss",
    device: Any = torch.device("cuda"),
    num_classes: int = 3,
    aggregation="TransMIL",
    n_heads: int = 16,
    ensemble: bool = False,
    source: str = "IMP",
    split: str = "val",
    trial: str = "trial"
    ):
    model.eval()
    if ensemble:
        n_heads = 1
        probas = torch.zeros(1, 3)
    metrics = {}
    for i in range(n_heads):
        if not aggregation == "MIL":
            pred = []
            true = []
        else:
            pred = torch.tensor([], device=device)
            true = torch.tensor([], device=device)
        yhat = torch.tensor([], device=device)

        auroc = AUROC(task='multiclass', num_classes=3, average=None)
        roc = ROC(task="multiclass", num_classes=3)
        confmat = ConfusionMatrix(task="multiclass", num_classes=3, normalize="true")

        for step, batch in enumerate(tqdm(dataloader)):
            
            if not aggregation == "MIL":
                h, target = batch
                h = h.to(device)
                target = target.to(device)

            with torch.cuda.amp.autocast():
                with torch.no_grad():   
                    
                    if not aggregation == "MIL":
                        out = model(h)
                    else:
                        probas, preds, targets = topk_instances(model, batch, topk=1, device=device, mode="eval")
                    if not aggregation == "MIL":
                        if ensemble:
                            for p in out["probas"]:
                                probas += p
                            probas /= len(out["probas"])
                        else:
                            probas = out["probas"][i]
                        pred.append(torch.argmax(probas, dim=1))
                        true.append(target)
                        yhat = torch.cat((yhat, probas))
                    else:
                        pred = torch.cat((pred, preds))
                        true = torch.cat((true, targets))
                        yhat = torch.cat((yhat, probas))
        if not aggregation == "MIL":  
            pred = torch.flatten(torch.stack(pred))
            true = torch.flatten(torch.stack(true))
        else:
            pred = torch.flatten(pred).to(torch.int64)
            true = torch.flatten(true).to(torch.int64)

        qwk = quadratic_weighted_kappa(pred.cpu(), true.cpu(), 0, num_classes-1)
        acc = sum(pred == true)/len(true)

        binary_pred = (pred > 0).to(torch.int16)
        binary_true = (true > 0).to(torch.int16)
        bin_acc = sum(binary_pred == binary_true)/len(binary_true)

        CM = confusion(binary_true.cpu().numpy(), binary_pred.cpu().numpy())

        tn=CM[0][0]
        tp=CM[1][1]
        fp=CM[0][1]
        fn=CM[1][0]
            
        sensitivity = tp/(tp + fn + 1e-6)
        specificity = tn/(tn + fp + 1e-6)
        precision = tp/(tp + fp + 1e-6)

        f1 = (2*precision*specificity)/(precision+specificity)

        multiclass_confmat = confmat(pred.cpu(), true.cpu())
        df_cm = pd.DataFrame(
            np.array(multiclass_confmat),
            columns=np.unique(true.cpu()),
            index = np.unique(true.cpu())
            )
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'

        auc = auroc(yhat, true)
        fpr, tpr, _ = roc(yhat, true)
        if split =="test":
            if not os.path.exists(
            os.path.join("saved",
                        aggregation,
                        trial,
                        run_id,
                        "plots"
                        )):
                os.makedirs(
                    os.path.join("saved",
                            aggregation,
                            trial,
                            run_id,
                            "plots"
                        )
                )
                
            c = ["m", "b", "y"]
            labels = ["non-neoplastic-vs-rest", "low-grade-vs-rest", "high-grade-vs-rest"]
                
            plt.clf()
            plt.figure(figsize = (9, 9))
            cmap = sns.cubehelix_palette(light=1, as_cmap=True)

            sns.heatmap(df_cm, cbar=False, annot=True, cmap=cmap, square=True, fmt='.3f',
                    annot_kws={'size': 10})
            plt.title('Multiclass Confusion Matrix')
            plt.savefig(
                os.path.join("saved",
                            aggregation,
                            trial,
                            run_id,
                            "plots",
                            "multiclass-confmat("+str(i)+").png"
                            ))
            plt.clf()
            for j in range(3):
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.plot(fpr[j].cpu(), tpr[j].cpu(), c[j], label = labels[j]+' (AUC = %0.3f)' % auc[j])
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'k--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Sensitivity (True Positive Rate)')
            plt.xlabel('Specificity (False Positive Rate)')
            plt.savefig(
                os.path.join("saved",
                            aggregation,
                            trial,
                            run_id,
                            "plots",
                            "roc-curve("+str(i)+").png"
                            ))

        
        m = {
            "qwk: ": np.around(qwk, decimals=4, out=None).item(),
            "acc": np.around(acc.cpu(), decimals=4, out=None).item(),
            "bin_acc": np.around(bin_acc.cpu(), decimals=4, out=None).item(),
            "sens": np.around(sensitivity, decimals=4, out=None).item(),
            "spec": np.around(specificity, decimals=4, out=None).item(),
            "prec": np.around(precision, decimals=4, out=None).item(),
            "f1": np.around(f1, decimals=4, out=None).item(),
            "auc-1-vs-rest": np.around(auc[0].cpu(), decimals=4, out=None).item(),
            "auc-2-vs-rest": np.around(auc[1].cpu(), decimals=4, out=None).item(),
            "auc-3-vs-rest": np.around(auc[2].cpu(), decimals=4, out=None).item(),
        }
        metrics[str(i)]=m
   
    with open(
        os.path.join(
            "saved",
            aggregation,
            trial,
            run_id,
            "metrics-"+split+"-"+source+".json"
        ), "w") as f:
            json.dump(metrics, f)


def main(
        num_classes: int = 3,
        batch_size: int = 1,
        workers: int = 8,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    args = parse_args()
    
    assert args.aggregation in ["CLAM_SB", "CLAM_MB", "TransMIL", "MIL"], "invalid value for 'aggregation'"
    
    testset = WSIEmbeddingsDataset(path=args.path, jsonfile=args.test_json, libraryfile=args.lib)
 
    g = torch.Generator()
    g.manual_seed(SEED)

    collate_fn = None
    if args.aggregation == "TransMIL":
        model = TransMIL(num_classes=num_classes, size=args.fc_size, n_heads=args.n_heads)
    elif args.aggregation == "CLAM_SB":
            model = CLAM_SB(
                n_classes=num_classes,
                size=[args.fc_size, args.fc_size, 256]
            )
    elif args.aggregation == "CLAM_MB":
        raise NotImplementedError
    else:
        model = torch.nn.Linear(args.fc_size, num_classes)
        collate_fn = custom_collate_fn


    ckpt = os.path.join(
        "saved",
        args.aggregation,
        args.trial,
        args.run_id,
        args.run_id+"_"+args.ckpt
    )
    
    model = model.to(device)
    model.load_state_dict(
            torch.load(ckpt, map_location=device)
    )
    # model.load_state_dict(
    #    torch.load(
    #        "/nas-ctm01/homes/jdfernandes/cadpath-ssl/saved/TransMIL/10-13-2023-13:33:16_transMIL-resnet34_single_best.pth",
    #        map_location=device
    #    )
    # )
    # model.cls_heads = torch.nn.ModuleList([
    #    torch.nn.Linear(args.fc_size, num_classes) for _ in range(args.n_heads)]).to(device)

    dataloader = DataLoader(
        testset,
        drop_last=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )
 
    test(
        model,
        dataloader,
        args.run_id,
        device=device,
        aggregation=args.aggregation,
        n_heads=args.n_heads,
        ensemble=args.ensemble,
        source=args.source,
        split=args.split,
        trial=args.trial,
        )
    print("Execution finished!")

if __name__ == '__main__':
    main()
