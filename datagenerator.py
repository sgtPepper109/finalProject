import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed = 24
batch_size = 16
n_classes = 2


def preprocess_data(img, mask, num_class):
    # Scale images
    img = img / 255.
    # Convert mask to one-hot
    labelencoder = LabelEncoder()
    n, h, w, c = mask.shape
    mask = mask.reshape(-1, 1)
    mask = labelencoder.fit_transform(mask)
    mask = mask.reshape(n, h, w, c)
    mask = to_categorical(mask, num_class)
    return (img, mask)


def trainGenerator(train_img_path, train_mask_path, num_class):
    img_data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(128, 128),
        batch_size=batch_size,
        seed=seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode=None,
        color_mode='grayscale',
        target_size=(128, 128),
        batch_size=batch_size,
        seed=seed
    )

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)

parent_dir = os.getcwd()

train_img_path = 'C:/Users/Acer/PycharmProjects/pythonProject/divided_data/train_images/'
train_mask_path = 'C:/Users/Acer/PycharmProjects/pythonProject/divided_data/train_masks/'

print('\n')
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=2)
print('\n')

val_img_path = 'C:/Users/Acer/PycharmProjects/pythonProject/divided_data/val_images/'
val_mask_path = 'C:/Users/Acer/PycharmProjects/pythonProject/divided_data/val_masks/'
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=2)

x, y = train_img_gen.__next__()

# plot examples for test
# for i in range(0, 3):
#     image = x[i,:,:,0]
#     mask = np.argmax(y[i], axis=2)
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.imshow(mask, cmap='gray')
#     plt.show()
