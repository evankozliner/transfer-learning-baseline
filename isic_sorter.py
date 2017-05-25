import pandas as pd
import shutil

DATA_DIR = "ISIC-2017_Training_Data/"
OUTPUT_DIR = "isic_data/"

def main():
    gt = pd.read_csv("ISIC-2017_Training_Part3_GroundTruth.csv")
    
    for row in gt.iterrows():
        data = row[1]
        if data['melanoma'] == 0.0:
            path = OUTPUT_DIR + "Benign/"
        else:
            path = OUTPUT_DIR + "Malignant/"
        shutil.copy(DATA_DIR + data['image_id'] + ".jpg", path)

if __name__ == "__main__":
    main()
