import os
import cv2
import random
import shutil
import splitfolders
import matplotlib.pyplot as plt

parent_dir = os.getcwd() # current project path

input_folder = os.path.join(parent_dir, 'data/')
output_folder = os.path.join(parent_dir, 'divided_data/')

os.mkdir(output_folder)

train_ratio, test_ratio, val_ratio = .7, .15, .15
print('\ntrain_ratio:', train_ratio, ', test_ratio:', test_ratio, ', val_ratio:', val_ratio, '\n')
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=42,
    ratio=(train_ratio, test_ratio, val_ratio),
    group_prefix=None
)

print('Splitfolders folder structure:')
print('divided_data/')
print('├───test')
print('│   ├───images')
print('│   └───masks')
print('├───train')
print('│   ├───images')
print('│   └───masks')
print('└───val')
print('    ├───images')
print('    └───masks')
#
print('\nCreating proper folder structure i.e.')
print('divided_data/')
print('├───test_images/')
print('│   └───test/')
print('├───test_masks/')
print('│   └───test/')
print('├───train_images/')
print('│   └───train/')
print('├───train_masks/')
print('│   └───train/')
print('├───val_images/')
print('│   └───val/')
print('└───val_masks/')
print('    └───val/')

print('\n')


def createDirectory(path):
    os.mkdir(path)
    print('Created Directory:', path)


createDirectory(os.path.join(output_folder, 'train_images/'))
createDirectory(os.path.join(output_folder, 'train_masks/'))
createDirectory(os.path.join(output_folder, 'train_images/train/'))
createDirectory(os.path.join(output_folder, 'train_masks/train/'))
createDirectory(os.path.join(output_folder, 'test_images/'))
createDirectory(os.path.join(output_folder, 'test_masks/'))
createDirectory(os.path.join(output_folder, 'test_images/test/'))
createDirectory(os.path.join(output_folder, 'test_masks/test/'))
createDirectory(os.path.join(output_folder, 'val_images/'))
createDirectory(os.path.join(output_folder, 'val_masks/'))
createDirectory(os.path.join(output_folder, 'val_images/val/'))
createDirectory(os.path.join(output_folder, 'val_masks/val/'))

# copy all images to the respective directories
train_path = os.path.join(output_folder, 'train/')
test_path = os.path.join(output_folder, 'test/')
val_path = os.path.join(output_folder, 'val/')


def copyAllImages(source, destination):
    files = os.listdir(source)
    print('Copying images from', source, 'to', destination)
    for file in files:
        shutil.copy(os.path.join(source, file), destination)

print('\n')
copyAllImages(train_path + 'images/', os.path.join(output_folder, 'train_images/train/'))
copyAllImages(train_path + 'masks/', os.path.join(output_folder, 'train_masks/train/'))
copyAllImages(test_path + 'images/', os.path.join(output_folder, 'test_images/test/'))
copyAllImages(test_path + 'masks/', os.path.join(output_folder, 'test_masks/test/'))
copyAllImages(val_path + 'images/', os.path.join(output_folder, 'val_images/val/'))
copyAllImages(val_path + 'masks/', os.path.join(output_folder, 'val_masks/val/'))

# remove redundant directories
print('Removing redundant directories')
shutil.rmtree(os.path.join(output_folder, 'train/'))
shutil.rmtree(os.path.join(output_folder, 'test/'))
shutil.rmtree(os.path.join(output_folder, 'val/'))

# test whether the images and masks match
train_img_dir = os.path.join(output_folder, 'train_images/train/')
train_mask_dir = os.path.join(output_folder, 'train_masks/train/')

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)
img_list.sort()
msk_list.sort()

num_images = len(img_list)
incorrect = False
if len(img_list) == len(msk_list):
    print("Total number of training images: ", num_images)

    for i in range(num_images):
        x, y = img_list[i], msk_list[i]
        # images are named in the format: 123_sat.jpg
        # masks are named in the format: 123_mask.png
        x = x[0:-8]
        y = y[0:-9]
        if x != y:
            incorrect = True
            print('Some Images are missing or incorrect')
            break

if not incorrect:
    print('\nImages and their masks match\n')

# plot images and check if image and mask are matching
img_num = random.randint(0, num_images - 1)
img_for_plot = cv2.imread(train_img_dir + img_list[img_num], 0)
mask_for_plot = cv2.imread(train_mask_dir + msk_list[img_num], 0)

plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(img_for_plot, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(mask_for_plot, cmap='gray')
plt.title('Mask')
plt.show()
