""" Builds a simple whole image directory for the transfer learning script to sanity-check the problem """

import pandas as pd
import shutil

DATA_DIR = 'final/'
OUTPUT_DIR = 'whole-img-dataset/'

if __name__ == "__main__":
    gt = pd.read_csv('final.csv')
    for i, row in gt.iterrows():
        if row['melanoma']:
            shutil.copy(DATA_DIR + row['image_id'] + '.jpg', OUTPUT_DIR + 'Malignant/')
        else
            shutil.copy(DATA_DIR + row['image_id'] + '.jpg', OUTPUT_DIR + 'Benign/')
            


