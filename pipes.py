import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#FILTER OUT CORRUPTED IMAGES
import os
dataset_path = "data/train_small/"

num_skipped = 0
for folder_name in ("crack", "nocrack"):
    folder_path = os.path.join(dataset_path, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images " % num_skipped)

#GENERATE A DATASET
image_size = (480, 720)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

#VISUALIZE THE DATA
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

#APPLYING DATA AUGMENTATION
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

#VISUALIZE AFTER AUGMENTATION
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()

#STANDARDIZING THE DATA#
augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

#Let's make sure to use buffered prefetching so we 
# can yield data from disk without having I/O 
# becoming blocking:

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

#BUILD A MODEL
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

print("Model Summary:")
#f_summary = open("model-summary.txt", "w")
#f_summary.write(model.summary())
#f_summary.close()

#TRAIN THE MODEL
epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)


print('HISTORY KEYS: ', history.history.keys())
#PLOT ACCURACY
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.show()

#PLOT LOSS
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')
#plt.show()

#WRITE ACC DATA TO FILE
f_hist = open("accuracy.txt", "a")
f_hist.write("\n\n")
f_hist.write('[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in history.history['accuracy']])))
f_hist.write("\n")
f_hist.write('[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in history.history['val_accuracy']])))
f_hist.write("\n")
f_hist.write('[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in history.history['loss']])))
f_hist.write("\n")
f_hist.write('[{:s}]'.format(', '.join(['{:.2f}'.format(x) for x in history.history['val_loss']])))
f_hist.write("\n")
f_hist.close()


#RUN INFERENCE ON NEW DATA
f = open("accuracy-data.csv", "w")
f.write("file,crack,nocrack\n")
for fname in os.listdir("data/test/"):
    test_image = "data/test/"+fname
    img = keras.preprocessing.image.load_img(
    test_image, target_size=image_size
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    score = predictions[0]
    print(test_image + "This image is %.2f percent crack and %.2f percent nocrack."
        % (100 * (1 - score), 100 * score))
    f.write(test_image+","+str(100 * (1 - score))+","+ str(100 * score)+"\n")

f.close()
