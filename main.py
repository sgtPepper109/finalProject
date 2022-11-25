import os
import data
import datagenerator
import unet

num_train_images = len(os.listdir(data.train_img_dir))
num_val_images = len(os.listdir(os.path.join(data.output_folder, 'val_images/val/')))

steps_per_epoch = num_train_images // datagenerator.batch_size
val_steps_per_epoch = num_val_images // datagenerator.batch_size
print('Steps per epoch: ', steps_per_epoch)
print('Validation steps per epoch: ', val_steps_per_epoch)

IMG_HEIGHT = datagenerator.x.shape[1]
IMG_WIDTH = datagenerator.x.shape[2]
IMG_CHANNELS = datagenerator.x.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
print('\nInput Shape: ', input_shape)

model = unet.build_unet(input_shape, n_classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print('\n\n')
history = model.fit(
    datagenerator.train_img_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=12,
    verbose=1,
    validation_data=datagenerator.val_img_gen,
    validation_steps=val_steps_per_epoch
)

model.save(os.path.join(data.parent_dir, 'saved.hdf5'))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()