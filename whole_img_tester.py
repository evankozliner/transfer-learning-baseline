from __future__ import division
import whole_img_classifier as clf
import os
import pandas as pd

# TODO Try held out test dir
WHOLE_IMG_DIR = "data_whole/"
GROUND_TRUTH = "ground_truth_whole.csv"

def main():
    total_correct = 0
    # Don't count skips
    total_count = 0
    ground_truth = pd.read_csv(GROUND_TRUTH)

    clf.load_model()

    for row in ground_truth.iterrows():
        data = row[1]
        img_id, ext = data['image_id'].split(".")
        ext = "." + ext
        if not (img_id + ext).lower() in os.listdir(WHOLE_IMG_DIR):
            continue
        total_count += 1
        predicted_class, perc = clf.classify(WHOLE_IMG_DIR + img_id.lower(), ext)
        total_correct += check_correctness(predicted_class, data['melanoma'])
    print "Accuracy: " + str(total_correct / total_count)
        
def check_correctness(predicted_class, numeric_class):
    numeric_prediction = 0 if predicted_class == 'benign' else 1
    if numeric_prediction == numeric_class:
        return 1
    return 0

if __name__ == "__main__":
    main()
