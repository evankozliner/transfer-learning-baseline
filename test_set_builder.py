from __future__ import division

import pandas as pd
import math

EXT = '.jpg'
TEST_SIZE = .2

def main():
    gt = pd.read_csv("final.csv")
    gt['image_id'] = gt['image_id'] + '.jpg'
    total = len(gt)
    benign = gt[gt.melanoma == 0]
    malignant = gt[gt.melanoma == 1]
    perc_benign = len(benign) / total
    perc_malignant = len(malignant) / total

    test_mal_count = int(math.floor(perc_malignant * total * TEST_SIZE))
    test_ben_count = int(math.floor(perc_benign * total *TEST_SIZE))

    test_mal = malignant.sample(test_mal_count )
    test_ben = benign.sample(test_ben_count)

    test_set = pd.concat([test_mal, test_ben])

    train_set = gt.drop(test_set.index)

    test_set.to_csv("test.csv")
    train_set.to_csv("train.csv")

    print test_set
    return test_set, train_set
    

if __name__ == "__main__":
    main()
