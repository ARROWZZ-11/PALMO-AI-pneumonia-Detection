
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------------
# 1. PROJECT SETTINGS
# --------------------------------

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --------------------------------
# 2. CHECK DATASET
# --------------------------------

for directory in [TRAIN_DIR, TEST_DIR]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Folder not found: {directory}"
        )

    for class_name in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(directory, class_name)

        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Missing folder: {path}"
            )

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

class_names = ["NORMAL", "PNEUMONIA"]
image_counts = {}

for class_name in class_names:
    class_path = os.path.join(TRAIN_DIR, class_name)

    image_counts[class_name] = sum(
        1 for filename in os.listdir(class_path)
        if filename.lower().endswith(valid_extensions)
    )

print("Class names:", class_names)
print("Image counts:", image_counts)

# --------------------------------
# 3. CLASS WEIGHTS
# --------------------------------

total_images = sum(image_counts.values())
class_weights = {}

for index, class_name in enumerate(class_names):

    count = image_counts[class_name]

    if count == 0:
        raise ValueError(
            f"No images found in {class_name}"
        )

    class_weights[index] = total_images / (
        len(class_names) * count
    )

print("Class weights:", class_weights)

# --------------------------------
# 4. LOAD DATASET
# --------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False
)

print("Training classes:", train_ds.class_names)

if train_ds.class_names != class_names:
    raise ValueError("Dataset class order mismatch!")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------
# 5. DATA AUGMENTATION
# --------------------------------

data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.02),
    layers.RandomZoom(0.05)

])

# --------------------------------
# 6. BUILD CNN MODEL
# --------------------------------

model = models.Sequential([

    layers.Input(shape=(180, 180, 1)),

    layers.Rescaling(1.0 / 255),

    data_augmentation,

    layers.Conv2D(32, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3, 3), padding="same"),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(1, activation="sigmoid")

])

# --------------------------------
# 7. COMPILE MODEL
# --------------------------------

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

)

model.summary()

# --------------------------------
# 8. CALLBACKS
# --------------------------------

checkpoint = ModelCheckpoint(
    "pneumonia_cnn_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# --------------------------------
# 9. TRAIN MODEL
# --------------------------------

print("\nStarting CNN training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stop
    ]

)

# --------------------------------
# 10. SAVE CLASS NAMES
# --------------------------------

with open("class_names.json", "w") as file:
    json.dump(class_names, file)

# --------------------------------
# 11. EVALUATE MODEL
# --------------------------------

print("\nEvaluating on test dataset...\n")

best_model = tf.keras.models.load_model(
    "pneumonia_cnn_model.keras"
)

results = best_model.evaluate(
    test_ds,
    return_dict=True
)

print("\nTest Results:")

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# --------------------------------
# 12. CONFUSION MATRIX
# --------------------------------

true_labels = np.concatenate([
    labels.numpy().ravel()
    for _, labels in test_ds
]).astype(int)

predicted_scores = best_model.predict(
    test_ds,
    verbose=1
).ravel()

predicted_labels = (
    predicted_scores >= 0.5
).astype(int)

matrix = tf.math.confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes=2
)

print("\nConfusion Matrix:")
print(matrix.numpy())

print("\nClass order:", class_names)

# --------------------------------
# 13. SAVE MODEL INFORMATION
# --------------------------------

print("\nCNN training completed successfully!")

print("Saved model: pneumonia_cnn_model.keras")
print("Saved classes: class_names.json")
