from __future__ import division

import subprocess
import cv2
import numpy as np
import math

BLOCK_SIZE = 64
THRESHOLD = .7

def main():
    img = 'Acl221'
    ext = ".jpg"
    
    # Assuming precompiled threshold_fusion
    # Run threshold fusion on raw image
    threshold_fusion_cmd = "./threshold_fusion " + img + ext

    # Probably won't work everywhere --be sure to compile TF first
    subprocess.call(threshold_fusion_cmd, shell = True)
    
    # Segment the image through python (Or matlab) into BLOCK_SIZExBLOCK_SIZE blocks
    thresholded_img_name = "out_" + img + "_bin.bmp"

    # Consider gaussian blur
    threshold_img = cv2.imread(thresholded_img_name, 0)
    original_img = cv2.imread(img + ext)

    width, height = threshold_img.shape

    x_steps = int(math.floor(width / BLOCK_SIZE))
    y_steps = int(math.floor(height / BLOCK_SIZE))

    # Crop image into BLOCK_SIZExBLOCK_SIZE squares
    # Consider padding with 0s as well
    image_blocks = np.zeros((x_steps * y_steps,BLOCK_SIZE,BLOCK_SIZE))
    for x in range(x_steps):
        for y in range(y_steps):
            img_block = threshold_img[
                    BLOCK_SIZE*x:BLOCK_SIZE*(x+1), 
                    BLOCK_SIZE*y:BLOCK_SIZE*(y+1)]
            print x + y * y_steps
            image_blocks[y + x * x_steps] = img_block
    
    #Filter blocks to have > 70% lesion
    def threshold_filter(block_idx):
        return np.sum(image_blocks[block_idx]) / 255 > BLOCK_SIZE **2 * THRESHOLD

    blocks_past_threshold = image_blocks[
            filter(threshold_filter, range(image_blocks.shape[0]))]

    return blocks_past_threshold
    # Classify each block
    
    # Vote on original image
    
    
if __name__ == "__main__":
    main()
