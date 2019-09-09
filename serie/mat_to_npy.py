# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path")
parser.add_argument("key")
parser.add_argument("--output_file", default="output.npy")

args = parser.parse_args()

path_image = args.image_path
key = args.key 

image = scipy.io.loadmat(path_image)[key].T
nRow  = scipy.io.loadmat(path_image)['nRow'][0][0]
nCol  = scipy.io.loadmat(path_image)['nCol'][0][0]
image = np.transpose(image.reshape(nCol, nRow, image.shape[1]), (1,0,2))
image = image.astype(np.float32)
np.save(args.output_file, image)