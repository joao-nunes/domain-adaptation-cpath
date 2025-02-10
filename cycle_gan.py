from typing import Any
from models.unet import UNet
from argparse import ArgumentParser
import torch.optim as optim
import torch
import torch.nn as nn
from cpathutils.datasets import TilesDataset
from utils.seeds import seed_worker
from torch.utils.data import   Dataset, DataLoader, Subset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from torch.optim.swa_utils import AveragedModel
from torch import autograd
import copy
from PIL import Image
import torchvision.models as models
from utils.identity import Identity
from models.transmil import TransMIL
import os 
from utils.qwk import *
import sys
import json

class ConvolutionalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        add_activation: bool = True,
        **kwargs
    ):
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            ConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvInstanceNormLeakyReLUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        Class object initialization for Convolution-InstanceNorm-LeakyReLU layer

        We use leaky ReLUs with a slope of 0.2.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(
        self, img_channels: int, num_features: int = 64, num_residuals: int = 9
    ):

        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.downsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features, 
                    num_features * 2,
                    is_downsampling=True, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.upsampling_layers = nn.ModuleList(
            [
                ConvolutionalBlock(
                    num_features * 4,
                    num_features * 2,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                ConvolutionalBlock(
                    num_features * 2,
                    num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.last_layer = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.downsampling_layers:
            x = layer(x)
        x = self.residual_layers(x)
        for layer in self.upsampling_layers:
            x = layer(x)
        return torch.tanh(self.last_layer(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvInstanceNormLeakyReLUBlock(
                    in_channels, 
                    feature, 
                    stride=1 if feature == features[-1] else 2,
                )
            )
            in_channels = feature
 
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        return self.model(x) # torch.sigmoid(self.model(x))

def parse_args():

    parser = ArgumentParser()
    parser.add_argument("--bern_indices", type=str, default="./data/train_test_splits/cycle_gan_split_bern_train.npy")
    parser.add_argument("--imp_indices", type=str, default="./data/train_test_splits/cycle_gan_split_imp_train.npy")
    parser.add_argument("--resnet_path", type=str, default="./saved/baseline/11-15-2024-15:57:02_resnet34_best.pth")
    parser.add_argument("--transmil_path", type=str, default="./saved/TransMIL/train-val-IMP/12-04-2024-10:13:28/12-04-2024-10:13:28_transMIL-resnet34_single_best.pth")
    parser.set_defaults()
    args = parser.parse_args()
    return args

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def main():

    SEED = 42
    LEARNING_RATE = 1e-4
    LAMBDA_CYCLE = 10
    LAMBDA_R1 = 10
    LAMBDA_IDENTITY = 5
    NUM_EPOCHS = 200 
    DEVICE = "cuda"
    BATCH_SIZE = 1
    EMA_DECAY = 0.999
    SAVE_EVERY = 10
    

    args = parse_args()
    writer = SummaryWriter()
    
    Disc1 = Discriminator(in_channels=3).to(DEVICE)
    Disc2 = Discriminator(in_channels=3).to(DEVICE)
    Gen1 = Generator(img_channels=3, num_residuals=15).to(DEVICE)
    Gen2 = Generator(img_channels=3, num_residuals=15).to(DEVICE)
    
    l1 = nn.L1Loss()
    gan_loss = nn.MSELoss() # nn.BCEWithLogitsLoss()
    
    # use Adam Optimizer for both generator and discriminator
    opt_disc = optim.Adam(
        list(Disc1.parameters()) + list(Disc2.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )

    opt_gen = optim.Adam(
        list(Gen2.parameters()) + list(Gen1.parameters()),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
    )
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    
    transform = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]) 
    imp_dataset = TilesDataset(transform=transform, N=1)
    
    bern_dataset = TilesDataset(path="CRC_Bern/tiles/level-0/", transform=transform, N=1)
    
    indices = np.load(args.imp_indices)
    imp_dataset = Subset(imp_dataset, indices=indices)

    indices = np.load(args.bern_indices)
    bern_dataset = Subset(bern_dataset, indices=indices) 

    g = torch.Generator()
    g.manual_seed(SEED)
    
    trainloader1 = DataLoader(
        imp_dataset,
        drop_last=True,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    trainloader2 = DataLoader(
        bern_dataset,
        drop_last=True,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        )
    d_losses = []
    g_losses = []
    gan_loss_G = []
    identity_loss = []
    cycle_loss = []

    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
        EMA_DECAY * averaged_model_parameter + (1-EMA_DECAY) * model_parameter
    ema_GEN1_model = AveragedModel(Gen1, device=DEVICE, avg_fn=ema_avg_fn)
    ema_GEN2_model = AveragedModel(Gen2, device=DEVICE, avg_fn=ema_avg_fn)
   
    # load resnet34
    model = models.resnet34(num_classes=3)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(args.resnet_path, map_location=DEVICE))
    model.fc = Identity()
    


    aggregator = TransMIL(num_classes=3, size=512, n_heads=1)
    aggregator = aggregator.to(DEVICE)
    aggregator.load_state_dict(torch.load(args.transmil_path, map_location=DEVICE))
    

    class TilesDataset_(Dataset):

        def __init__(
        self,
        root="/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/",
        path="CRC_Bern/tiles/level-0/",
        libfile="./data/BernCRC_test_otsu_512_100_map2.tar",
        jsonfile="./data/train_test_splits/bern_val_split_fold_1.json",
        return_key: bool = True,
        ):
            
            super(TilesDataset_, self).__init__()
                
            
            self.transform = A.Compose([
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ])

            
            self.image_ids=[]            
            for subdir in os.listdir(os.path.join(root, path)):
                files = os.listdir(os.path.join(root, path, subdir))
                files.sort()
                for file in files:
                    self.image_ids.append(os.path.join(root, path, subdir, file))
            with open(jsonfile, "r") as f:
                splits= json.load(f)
                splits = [v.split(".")[0] for v in splits]
            image_ids_=[]
            for v in splits:
                for image_id in self.image_ids:
                    if v in image_id:
                        image_ids_.append(image_id)
            self.image_ids = image_ids_

            lib = torch.load(libfile)
            slide_names=[s.split(".")[0].split("/")[-1] for s in lib["slides"]]
            self.targets = {k: v for k,v in zip(slide_names, lib["slide_target"])}
    
            self.return_key = return_key

            
        def __getitem__(self,index):
        
            img = Image.open(self.image_ids[index]).convert("RGB")
            img = np.array(img)
    
            key = self.image_ids[index].split(".")[0].split("/")[-1]
            k = key.split("-")[0]

            if self.transform is not None:
                img = self.transform(image=img)["image"]
        
        
            target = torch.tensor(self.targets[k], dtype=torch.int64)
            
            if self.return_key:
                return k, img, target
            return img, target
        
        def __len__(self):
            return len(self.image_ids)

    valset = TilesDataset_()
    valloader = DataLoader(
        valset,
        batch_size=24,
        num_workers=16,
        pin_memory=True
    )

    best_qwk = -1
    for epoch in range(NUM_EPOCHS):
    
        imp_reals = 0
        imp_fakes = 0
        
        tr2_iterator = iter(trainloader2)



        for step, data1 in enumerate(trainloader1):
            try:
                data2 = next(tr2_iterator)
            except StopIteration:
                tr2_iterator = iter(trainloader2)
                data2 = next(tr2_iterator)
            
            imp, _ = data1
            bern  = data2
            
            imp = imp.to(DEVICE)
            bern = bern.to(DEVICE)
    
            # train discriminators
            Gen1.train()
            with torch.cuda.amp.autocast():
                fake_imp = Gen1(bern)
            
                D_imp_real = Disc1(imp)
                
                D_imp_fake = Disc1(fake_imp.detach())
               
                imp_reals += D_imp_real.mean().item()
                imp_fakes += D_imp_fake.mean().item()
               
                D_imp_real_loss = gan_loss(D_imp_real, torch.ones_like(D_imp_real))
                D_imp_fake_loss = gan_loss(D_imp_fake, torch.zeros_like(D_imp_fake))
                D_imp_loss = D_imp_real_loss + D_imp_fake_loss # + LAMBDA_R1*reg1
               
                fake_bern = Gen2(imp)
               
                D_bern_real = Disc2(bern)
                D_bern_fake = Disc2(fake_bern.detach())
               
                D_bern_real_loss = gan_loss(D_bern_real, torch.ones_like(D_bern_real))
                D_bern_fake_loss = gan_loss(D_bern_fake, torch.zeros_like(D_bern_fake))
                D_bern_loss = D_bern_real_loss + D_bern_fake_loss # + LAMBDA_R1*reg2
               
               
                D_loss = (D_imp_loss + D_bern_loss) / 2
                d_losses.append(D_loss.mean().item())
            if step % 500 == 0:
                writer.add_scalar("D_loss", np.mean(d_losses), step)
                d_losses = []
                img = fake_imp.detach().cpu().numpy().squeeze()
                img = 255*(img - img.min())/(img.max()-img.min())
                img = np.moveaxis(img, 0, -1)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img.save("fake_imp_.png")

                img = fake_bern.detach().cpu().numpy().squeeze()
                img = 255*(img - img.min())/(img.max()-img.min())
                img = np.moveaxis(img, 0, -1)
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img.save("fake_bern_.png")
                

               
            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
           
            # train generators
            with torch.cuda.amp.autocast():
               # adversarial losses
               D_imp_fake = Disc1(fake_imp)
               D_bern_fake = Disc2(fake_bern)
               loss_G_imp = gan_loss(D_imp_fake, torch.ones_like(D_imp_fake))
               loss_G_bern = gan_loss(D_bern_fake, torch.ones_like(D_bern_fake))

               # cycle losses
               cycle_bern = Gen2(fake_imp)
               cycle_imp = Gen1(fake_bern)
               cycle_bern_loss = l1(bern, cycle_bern).mean()
               cycle_imp_loss = l1(imp, cycle_imp).mean()

               identity_imp = Gen1(imp)
               identity_bern = Gen2(bern)

               identity_imp_loss = l1(imp, identity_imp).mean()
               identity_bern_loss = l1(bern, identity_bern).mean()

               # total loss
               G_loss = (
                   loss_G_bern
                   + loss_G_imp
                   + cycle_bern_loss * LAMBDA_CYCLE
                   + cycle_imp_loss * LAMBDA_CYCLE
                   + identity_imp_loss * LAMBDA_IDENTITY
                   + identity_bern_loss * LAMBDA_IDENTITY
               )
               g_losses.append(G_loss.mean().item())
               gan_loss_G.append((loss_G_bern.mean().item()+loss_G_imp.mean().item())/2)
               cycle_loss.append((cycle_bern_loss.mean().item()+cycle_imp_loss.mean().item())/2)
               identity_loss.append((identity_bern_loss.mean().item()+identity_imp_loss.mean().item())/2)

            if step % 500 == 0:
                writer.add_scalar("G_loss", np.mean(g_losses), step)
                writer.add_scalar("gan_loss_G", np.mean(gan_loss_G), step)
                writer.add_scalar("cycle_loss", np.mean(cycle_loss), step)
                writer.add_scalar("identity_loss", np.mean(identity_loss), step)
                g_losses = []
                gan_loss_G = []
                cycle_loss = []
                identity_loss = []

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            ema_GEN1_model.update_parameters(Gen1)
            ema_GEN2_model.update_parameters(Gen2)
            
            if step % 5000 == 0 :


                # validate
                embeddings={}
                preds = []
                true = []
                Gen1.eval()

                for key, img, target in valloader:
                    img = img.to(DEVICE)
                    with torch.no_grad():
                        fake_imp = Gen1(img)
                        embs = model(fake_imp)
                    for i, z in enumerate(embs):
                        if not key[i] in list(embeddings.keys()):
                            if len(list(embeddings.keys())) % SAVE_EVERY == 0 and len(list(embeddings.keys()))!=0:
                                # transMIL
                                for _, emb in embeddings.items():
                                    with torch.no_grad():
                                        out=aggregator(emb.unsqueeze(0))
                                        probas = out["probas"][0]
                                    preds.append(torch.argmax(probas, dim=1))
                                    
                                embeddings={}

                            embeddings[key[i]] = z.unsqueeze(0)
                            true.append(target[i])
                        else:
                            embeddings[key[i]] = torch.cat((embeddings[key[i]], z.unsqueeze(0)))

                for _, emb in embeddings.items():
                    with torch.no_grad():
                        out=aggregator(emb.unsqueeze(0))
                        probas = out["probas"][0]
                    preds.append(torch.argmax(probas, dim=1))
                        
                        
                preds = torch.flatten(torch.stack(preds))
                true = torch.flatten(torch.stack(true))
                
                qwk = quadratic_weighted_kappa(preds.cpu(), true.cpu(), 0, 2)
                if qwk >= best_qwk:

                    best_qwk = qwk

                    torch.save(
                    {
                    "model_state_dict": Gen1.state_dict(),
                    "optim_state_dict": opt_gen.state_dict(),
                    }, "./saved/cycle-gan/Gen1_.pth")
                    
                    torch.save(
                    {
                    "model_state_dict": Gen2.state_dict(),
                    "optim_state_dict": opt_gen.state_dict(),
                    }, "./saved/cycle-gan/Gen2_.pth")
                    
                    torch.save(
                    {
                    "model_state_dict": Disc1.state_dict(),
                    "optim_state_dict": opt_disc.state_dict(),
                    }, "./saved/cycle-gan/Disc1_.pth") 
                    
                    torch.save(
                    {
                    "model_state_dict": Disc2.state_dict(),
                    "optim_state_dict": opt_disc.state_dict(),
                    }, "./saved/cycle-gan/Disc2_.pth")

                    torch.save(ema_GEN1_model.state_dict(), "./saved/cycle-gan/ema_Gen1_model_.pth")
                    torch.save(ema_GEN2_model.state_dict(), "./saved/cycle-gan/ema_Gen2_model_.pth")
                    


if __name__ == "__main__":

    main()

