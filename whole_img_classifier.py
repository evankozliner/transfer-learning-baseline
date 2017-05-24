from __future__ import division
from collections import Counter

import os
# Supress annoying build from source log messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import shutil
import tensorflow as tf
import subprocess
import cv2
import numpy as np
import math
import scipy.misc

BLOCK_SIZE = 64
THRESHOLD = .7
INTER_DIR = "inter_dir"
INTERMEDIATE_DIR_FORMAT = INTER_DIR + "/{0}.jpg"
GRAPH_FILENAME = "retrained_graph_64x64.pb"

def classify(img, ext):
    """ Takes in an image of a lesion and its extension and responds 
        with a classification for that image done by classification on 
        that lesion's blocks as well as the percentage that classification
        "won" with (by votes).
        Saves the blocks intermediately. This approach was only necessitated 
        because I was not sure how to get the necessary string based JPEG 
        format without first saving them as JPEGs. 
    """

    img = img.lower()
    ext = ext.lower()
    
    blocks_past_threshold = get_blocks_past_threshold(img,ext, BLOCK_SIZE)

    block_file_paths = save_blocks(blocks_past_threshold, INTER_DIR)

    most_common = load_model_and_vote(block_file_paths)[0]

    remove_temp_files(block_file_paths, img, ext)
    
    #print blocks_past_threshold.shape[0]
    #print most_common[0], most_common[1] / blocks_past_threshold.shape[0]
    return most_common[0], most_common[1] / blocks_past_threshold.shape[0]

def save_blocks(block_data, folder, build_folder=True, name=''):
    if build_folder: os.mkdir(folder)
    paths = []
    for block_idx in range(block_data.shape[0]):
        if build_folder:
            path = INTERMEDIATE_DIR_FORMAT.format(str(block_idx))
        else:
            path = folder + name + str(block_idx)  + '.jpg'
        #print path
        #print block_data[block_idx].shape
        scipy.misc.imsave(path, block_data[block_idx])
        paths.append(path)
    return paths

def remove_temp_files(paths, img, ext):
    """ Removes the temporarily saved jpegs """
    print "Deleting temporary images..."
    thresholded_img_name = "out_" + img + "_bin.bmp"
    nonbinary_threshold_img = "out_" + img + ".bmp"
    shutil.rmtree(INTER_DIR)
    os.remove(nonbinary_threshold_img)
    os.remove(thresholded_img_name)

def get_blocks_past_threshold(img, ext, block_size):
    # Assuming precompiled threshold_fusion
    # Run threshold fusion on raw image
    threshold_fusion_cmd = "./fourier_0.8/threshold_fusion " + img + ext

    print threshold_fusion_cmd
    # Probably won't work everywhere --be sure to compile TF first
    subprocess.call(threshold_fusion_cmd, shell = True)
    
    thresholded_img_name = "out_" + img + "_bin.bmp"

    threshold_img = cv2.imread(thresholded_img_name, 0)
    print img + ext
    original_img = cv2.imread(img + ext)

    # CV2 reads in BGR instead of RGB order for some u/k reason
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    width, height = threshold_img.shape

    x_steps = int(math.floor(width / block_size))
    y_steps = int(math.floor(height / block_size))

    # Crop image into block_sizexblock_size squares
    # Consider padding with 0s as well
    image_blocks = np.zeros((x_steps * y_steps,block_size,block_size))
    image_blocks_orig = np.zeros((x_steps * y_steps,block_size,block_size, 3))

    for x in range(x_steps):
        for y in range(y_steps):
            image_blocks[y + x * x_steps] = \
                    get_block_by_coordinates(x,y, threshold_img, block_size)
            image_blocks_orig[y + x * x_steps] = \
                    get_block_by_coordinates(x,y, original_img, block_size)
    
    #Filter blocks to have > 70% lesion
    # Probably could just pass the image instead of the index lol
    def threshold_filter(block_idx):
        return np.sum(image_blocks[block_idx]) / 255 > block_size **2 * THRESHOLD

    return image_blocks_orig[filter(threshold_filter, range(image_blocks.shape[0]))]

def get_block_by_coordinates(x, y, img, block_size):
    return img[
            block_size*x:block_size*(x+1), 
            block_size*y:block_size*(y+1)]

def load_model_and_vote(block_paths):
    """ Return the class of the given image blocks (numpy array) and the count of votes
        for the array"""
    image_data = []
    label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

    for path in block_paths:
        block_data = tf.gfile.FastGFile(path, 'rb').read()
        image_data.append(block_data)

    #load_model()

    with tf.Session() as sess:
        return vote_on_data(sess, image_data, label_lines)

def load_model():
    """ Loads the appriopriate graph as the default graph """
    # TODO be sure retrained_graph name is consistant w/ block size
    with tf.gfile.FastGFile(GRAPH_FILENAME, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def vote_on_data(sess, image_data, label_lines):
    votes = []
    for block_data in image_data:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = list(sess.run(softmax_tensor,
                {'DecodeJpeg/contents:0': block_data})[0])
        top_prediction = predictions.index(max(predictions))
        votes.append(label_lines[top_prediction])

    #TODO If it is even pick Mal
    return Counter(votes).most_common(1)
    
if __name__ == "__main__":
    load_model()
    img = 'ndl078'
    ext = ".jpg"
    classify(img, ext)
