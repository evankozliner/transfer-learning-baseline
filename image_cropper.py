import whole_img_classifier as threshold_utils

import pandas as pd
import os

WHOLE_IMG_DIR = "data_whole/"
BLOCK_SIZE = 128
OUTPUT_DIR = "data_{0}x{0}/".format(str(BLOCK_SIZE))
EXT = '.jpg'

def main():
    gt = pd.read_csv('ground_truth_whole.csv')

    for i, row in gt.iterrows():
        img = row['image_id'].lower()
        if img not in os.listdir(WHOLE_IMG_DIR): 
            continue
        name = img.split(".")[0]
        subdir = 'Malignant/' if row['melanoma'] else 'Benign/'
        blocks = threshold_utils.get_blocks_past_threshold(
                WHOLE_IMG_DIR + name, 
                EXT, 
                BLOCK_SIZE)
        block_file_paths = threshold_utils.save_blocks(
                blocks, OUTPUT_DIR + subdir, build_folder=False, name=name)

def build_output_dirs():
    os.mkdir(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR + "Benign")
    os.mkdir(OUTPUT_DIR + "Malignant")

if __name__ == "__main__":
    build_output_dirs()
    main()
