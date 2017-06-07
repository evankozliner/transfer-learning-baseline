import pandas as pd
import shutil

EXT = '.jpg'
DATA_DIR = 'final/'
DEST_TEST = 'test-whole-imgs/'
DEST_TRAIN = 'train-whole-imgs/'

if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    for i, row in train.iterrows():
        shutil.copy(DATA_DIR + row['image_id'], DEST_TRAIN)

    for i, row in test.iterrows():
        shutil.copy(DATA_DIR + row['image_id'], DEST_TEST)
        

    
