import datagenerator

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU

import warnings
warnings.filterwarnings('ignore')

parent_dir = os.getcwd()
model = load_model(parent_dir + '/saved.hdf5', compile=False)

test_img_path = parent_dir + '/divided_data/test_images/'
test_mask_path = parent_dir + '/divided_data/test_masks/'

test_img_gen = datagenerator.trainGenerator(test_img_path, test_mask_path, num_class=2)

test_image_batch, test_mask_batch = test_img_gen.__next__()

y_pred = model.predict(test_image_batch)
y_pred_argmax = np.argmax(y_pred,axis=3)
y_argmax = np.argmax(test_mask_batch, axis=3)


n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_argmax, y_pred_argmax)
print("Mean IOU = ", IOU_keras.result().numpy())

for i in range(11):
	test_img_number = random.randint(0, len(test_image_batch)-1)
	test_img = test_image_batch[test_img_number]
	ground_truth = test_mask_batch[test_img_number]
	ground_truth = np.argmax(ground_truth, axis=2)
	test_img_norm = test_img[:,:,0][:,:,None]

	test_img_input = np.expand_dims(test_img_norm, 0)
	prediction = (model.predict(test_img_input))
	predicted_img = np.argmax(prediction, axis=3)[0,:,:]

	plt.figure(figsize=(12, 8))
	plt.subplot(231)
	plt.title('Testing Image')
	plt.imshow(test_img[:,:,0], cmap='gray')
	plt.subplot(232)
	plt.title('Testing Label')
	plt.imshow(ground_truth, cmap='jet')
	plt.subplot(233)
	plt.title('Prediction on test image')
	plt.imshow(predicted_img, cmap='jet')
	plt.show()