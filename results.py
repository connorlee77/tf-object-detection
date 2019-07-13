import keras
from keras import backend as K

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

import tensorflow as tf
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

from utils import label_map_util

# for mscoco
# Images in input folder need to be named s.t they are pulled in time series order
def detect_images(func_detect_image, func_read_img, path_to_input_images, path_to_output_images=None, path_to_output_csv=None, output_PIL=True):
	
	PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
	labels_to_names = label_map_util.get_label_map_dict(PATH_TO_LABELS, use_display_name=True)

	image_filenames = [f for f in listdir(path_to_input_images) if isfile(join(path_to_input_images, f))]
	data = []
	image_filenames = sorted(image_filenames)

	for i, image_filename in tqdm(enumerate(image_filenames)):

		input_filename = os.path.join(path_to_input_images, image_filename)
		output_filename = os.path.join(path_to_output_images, 'processed_' + image_filename)

		try:
			image = func_read_img(input_filename)
		except:
			print('{} is not a valid image file'.format(image_filename))
			continue

		output_image, detection = func_detect_image(image)
		data.append(detection)

		if output_PIL:
			output_image.save(output_filename)
		else:
			plt.figure()
			plt.axis('off')
			plt.imsave(output_filename, output_image)
	
	output_csv = os.path.join(path_to_output_csv, 'results.csv')
	df = pd.DataFrame(np.array(data))
	df.to_csv(output_csv, header=list(labels_to_names.values()), index=False)


