from openslide import *
import os
import torch
from openslide import *
from typing import Tuple
from argparse import ArgumentParser
import multiprocessing
import json
import pandas as pd
from PIL import Image
import numpy as np



class HistoPrep():
    def __init__(self):
        self.version = "1.0.1"

    def extract(self, f, coords, outdir, level, train, source, size=(512, 512)):

        slide_id = f.split("/")[-1].split(".")[0]
        if not os.path.exists(
            os.path.join(outdir, "level-"+str(level), slide_id)):
                os.makedirs(
                    os.path.join(outdir, "level-"+str(level), slide_id))

        if source == "IMPDiagnostics":
            try:
                slide = open_slide(f.replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/"))
            except:
                if train:
                    slide = open_slide(f.replace("/home/imp-data/uploads/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500/"))
                else:
                    slide = open_slide(f.replace("/home/imp-data/uploads/non_annotated/", "/nas-ctm01/partners/IMPDIAGNOSTICS/cadpath/CRC/extra_500/"))
        else:
            slide = open_slide(f)
        for coord in coords:
            # if scale==1:
	    #    size_ = (size[0] + 2*size[0]*int(2**(level-1)), size[1] + 2*size[1]*int(2**(level-1)))
            #    coord = (coord[0]-size[0]*int(2**(level-1)), coord[1]-size[1]*int(2**(level-1)))

            image = slide.read_region(coord, level, size).convert('RGB')
            image.thumbnail(size, Image.ANTIALIAS)
            image.save(
                os.path.join(
                        outdir,
                        "level-"+str(level),
                        slide_id,
                        slide_id+"-"+str(coord[0])+"-"+str(coord[1])+".png",
                        ))

    def save_targets_from_json(self, libfile: str, jsonfile: str, outdir: str):

        with open(jsonfile, "r") as f:
            jsonf = json.load(f)
        lib = torch.load(libfile)

        slides = lib["slides"]
        grid = jsonf["grid"]
        tile_targets = []
        file_ids = []
        for i, slideIDX in enumerate(jsonf["slideIDX"]):
          slide_id = slides[slideIDX]
          coord = grid[i]
          file_ids.append(slide_id.split("/")[-1].split(".")[0]+"-"+str(coord[0])+"-"+str(coord[1]))
          tile_targets.append(jsonf["targets"][i])

        labels = {"img_id": file_ids, "target": tile_targets}
        df=pd.DataFrame(labels)
        df.to_csv(os.path.join(outdir, "targets.csv"))

    def pataki_filter(self, f):
        # preprocessing suggest in
        # HunCRC: annotated pathological slides to enhance deep learning applications in colorectal cancer screening 
        # Pataki et al.
        # https://www.nature.com/articles/s41597-022-01450-y
        img = Image.open(f).convert("RGB")
        img = np.array(img)
        S = 1 - np.min(img, -1)/(np.mean(img, -1) + 1e-8)
        I = np.mean(img, -1)

        if not ((np.sum(S <= 0.05)/(512*512) <= 0.5) and (np.sum(I >= 245)/(512*512) <= 0.5)):
            os.remove(f)
    
    def _is_pos_def(self, A):
      if np.allclose(A, A.T):
          try:
              np.linalg.cholesky(A)
              return True
          except np.linalg.LinAlgError:
              return False
      else:
          return False 
    
    def MahalanobisDist(self, data, verbose=False):
        covariance_matrix = np.cov(data, rowvar=False)
        if self._is_pos_def(covariance_matrix):
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            if self._is_pos_def(inv_covariance_matrix):
                vars_mean = []
                for i in range(data.shape[0]):
                    vars_mean.append(list(data.mean(axis=0)))
                diff = data - vars_mean
                md = []
                for i in range(len(diff)):
                    md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    
                if verbose:
                    print("Covariance Matrix:\n {}\n".format(covariance_matrix))
                    print("Inverse of Covariance Matrix:\n {}\n".format(inv_covariance_matrix))
                    print("Variables Mean Vector:\n {}\n".format(vars_mean))
                    print("Variables - Variables Mean Vector:\n {}\n".format(diff))
                    print("Mahalanobis Distance:\n {}\n".format(md))
                return md
            else:
                print("Error: Inverse of Covariance Matrix is not positive definite!")
        else:
            print("Error: Covariance Matrix is not positive definite!")
    
    def malahanobis_outlier_detect(self, datapath):
        lst=[]
        ids = os.listdir(datapath)
        for id_ in ids:
            df = pd.read_csv(os.path.join(datapath, id_))
            lst.append(df.to_numpy())
        data = np.vstack(lst)
        d_m = self.MahalanobisDist(data, verbose=True)
        return d_m

    def save_tiles_from_lib(self, libfile: str, outdir: str, size: Tuple=(512, 512), source="IMPDiagnostics", train=True, batch_size = 256):

        lib = torch.load(libfile)
        files = lib["slides"]
        grid = lib["grid"]
        level = lib["level"]
        multiprocessing.set_start_method("spawn")

        train = [train]*batch_size
        source = [source]*batch_size
        outdir = [outdir]*batch_size
        level = [level]*batch_size
        print(level)
        num_workers = 14
        print(f"Num workers: {num_workers}")
        for i in range(0, len(files), batch_size):

            tup_1 = (*files[i:i+batch_size],)
            tup_2 = (*grid[i:i+batch_size],)

            batch = list(zip(tup_1, tup_2, (*outdir,), (*level,), (*train,), (*source,)))

            with multiprocessing.Pool(processes=num_workers) as p:
                p.starmap(self.extract, batch)

    def process_artifacts(self, datapath, batch_size=256, num_workers=14):

        multiprocessing.set_start_method("spawn")
        imglist = []
        print("Loading files directory...")
        for root, dirs, files in os.walk(datapath, topdown=False):
            for file in files:
                imglist.append(os.path.join(root, file))
        print("Processing files...")

        for i in range(0, len(imglist), batch_size):
            batch = imglist[i:i+batch_size]
            with multiprocessing.Pool(processes=num_workers) as p:
                p.map(self.pataki_filter, batch)