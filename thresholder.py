""" Runs threshold fusion for each of images included in both the ISIC and dermoscopic image dataset """

import pandas as pd
import os
import shutil
import subprocess

DATA_DIR = 'final/'
OUT_DIR = 'thresholded_final/'
GROUND_TRUTH = 'final.csv'

def run_threshold_fusion(img_name, ext):
    threshold_fusion_cmd = "./fourier_0.8/threshold_fusion " + \
            DATA_DIR + \
            img_name + \
            ext
    subprocess.call(threshold_fusion_cmd, shell = True)
    return "out_" + img_name + "_bin.bmp"

if __name__ == "__main__":
    gt = pd.read_csv(GROUND_TRUTH)

    for i,row in gt.iterrows():
        thresholded_img = run_threshold_fusion(row['image_id'], '.jpg')



