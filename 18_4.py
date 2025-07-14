import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A  # dodajemy import albumentations

# Dodajemy własną implementację mish
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

tf.keras.utils.get_custom_objects().update({'mish': tf.keras.layers.Activation(mish)})

train, test = tf.keras.datasets.fashion_mnist.load_data()

# wydobycie obrazów oraz labelek
images, labels = train

# normalizacja wartości pikseli
images = (images/255.0).astype(np.float32)

# zapisujemy dane jako int
labels = labels.astype(np.int32)

X_train, X_test, y_train, y_test = \
    train_test_split(images, labels, test_size=0.1, random_state=10, stratify=labels)

# Przygotowanie transformacji za pomocą albumentations
transform = A.Compose([
    # A.Resize(32, 32),  # Slightly larger than target
    A.RandomCrop(28, 28, p=0.8),
    A.Rotate(limit=30, p=0.8),  # Small rotation
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.8),
    # A.CropAndPad(px=10, p=0.8),
    # A.Normalize(mean=[0.1307], std=[0.3081]),  # MNIST stats
    # A.ToTensorV2(),
])

def augment(image, label):
    # Konwertujemy obraz do formatu wymaganego przez albumentations
    image = tf.cast(image * 255, tf.uint8)
    
    # Definiujemy funkcję wykonującą augmentację na pojedynczym obrazie
    def apply_augmentation(img):
        img_np = img.numpy()
        augmented = transform(image=img_np)
        return augmented['image']
    
    # Używamy tf.py_function do wykonania operacji numpy w kontekście grafu TF
    augmented_image = tf.py_function(
        apply_augmentation,
        [image],
        tf.uint8
    )
    
    # Przywracamy kształt i normalizujemy
    augmented_image = tf.cast(augmented_image, tf.float32) / 255.0
    augmented_image.set_shape(image.shape)
    
    return augmented_image, label

# Przygotowanie datasetu
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000)

# Tworzymy dwa datasety - oryginalny i augmentowany
original_ds = train_ds.batch(16)
augmented_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

# Łączymy oba datasety
ds = tf.data.Dataset.concatenate(original_ds, augmented_ds)
ds = ds.shuffle(1000)

def custom_lr_schedule(epoch, current_lr):
    decay_rate = 0.8
    decay_step = 2
    min_lr = 0.00005
    if epoch % decay_step == 0 and epoch != 0:
        return max(min_lr, current_lr * decay_rate**(epoch/decay_step))
    return max(min_lr, current_lr)
    # initial_lr = 0.002
    # if epoch < 3:
    #     return initial_lr
    # elif epoch < 5:
    #     return initial_lr * 0.7
    # elif epoch < 7:
    #     return initial_lr * 0.4
    # else:
    #     return initial_lr * 0.1

# Model z L2 regularyzacją
f_mnist_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(1000, activation='mish'),
    Dense(300, activation='mish'),
    Dense(150, activation='mish'),
    Dense(50, activation='mish'),
    Dense(10, activation='softmax')
])

f_mnist_model.summary()

initial_lr = 0.0006
adam_optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_lr,
)

f_mnist_model.compile(
    optimizer=adam_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Przygotowanie validation dataset
val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(16)

train_stats = f_mnist_model.fit(
    ds,
    verbose=1,
    epochs=15,
    validation_data=val_ds,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        # tf.keras.callbacks.LearningRateScheduler(custom_lr_schedule)
    ]
)

f_mnist_model.save('f_mnist_model_v2.h5')

# Wyświetlanie wyników
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_stats.history['accuracy'], label='Training Accuracy')
plt.plot(train_stats.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_stats.history['loss'], label='Training Loss')
plt.plot(train_stats.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

