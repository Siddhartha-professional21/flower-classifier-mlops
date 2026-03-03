# ============================================================
# train.py - Flower Classifier using MobileNetV2
# ============================================================
# What this file does:
# 1. Loads flower images from dataset/flowers folder
# 2. Prepares/resizes images for the model
# 3. Builds MobileNetV2 model (pretrained, fast & accurate)
# 4. Trains the model on your flower images
# 5. Saves accuracy/loss graphs and confusion matrix
# 6. Tracks everything using MLflow
# 7. Saves the trained model to model/ folder
# ============================================================


# ---- STEP 1: Import all required libraries ----
import os                          # to work with file paths
import numpy as np                 # for number operations
import matplotlib.pyplot as plt    # for plotting graphs
import seaborn as sns              # for confusion matrix heatmap
import mlflow                      # for experiment tracking
import mlflow.keras                # to save keras model in mlflow

from sklearn.metrics import confusion_matrix, classification_report

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2          # pretrained model
from tensorflow.keras.models import Model                       # to build model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # loads images


# ============================================================
# STEP 2: Set basic settings (easy to change later)
# ============================================================

DATASET_PATH = "dataset/flowers"   # path to your flowers folder
IMAGE_SIZE   = (96, 96)            # resize all images to 96x96 pixels
BATCH_SIZE   = 32                  # how many images to process at once
EPOCHS       = 10                  # how many times to train on full dataset
NUM_CLASSES  = 5                   # daisy, dandelion, rose, sunflower, tulip

# Class names (same as your folder names, alphabetical order)
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


# ============================================================
# STEP 3: Load and Prepare Images
# ============================================================
# ImageDataGenerator does two things:
#   - rescale: converts pixel values from 0-255 → 0-1 (models work better this way)
#   - validation_split: automatically splits 80% train, 20% validation

print("\n[1/6] Loading images from dataset...")

# Training data generator (with data augmentation to improve accuracy)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # normalize pixels to 0-1
    validation_split=0.2,      # 20% images saved for validation
    rotation_range=20,         # randomly rotate images (augmentation)
    zoom_range=0.2,            # randomly zoom in/out
    horizontal_flip=True       # randomly flip images left-right
)

# Validation data generator (NO augmentation, only rescale)
val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# Load TRAINING images from folder
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,     # resize every image to 96x96
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # one-hot encoding for 5 classes
    subset='training',          # use 80% for training
    shuffle=True
)

# Load VALIDATION images from folder
val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',        # use 20% for validation
    shuffle=False               # don't shuffle validation (needed for confusion matrix)
)

print(f"    Training images   : {train_data.samples}")
print(f"    Validation images : {val_data.samples}")
print(f"    Classes found     : {list(train_data.class_indices.keys())}")


# ============================================================
# STEP 4: Build the Model using MobileNetV2
# ============================================================
# MobileNetV2 is a pretrained model (already learned from millions of images)
# We just add our own final layers on top for 5 flower classes
# This is called TRANSFER LEARNING - reusing someone else's learning!

print("\n[2/6] Building MobileNetV2 model...")

# Load MobileNetV2 base (without the final classification layer)
base_model = MobileNetV2(
    input_shape=(96, 96, 3),   # 96x96 pixels, 3 = RGB color channels
    include_top=False,          # remove original classification head
    weights='imagenet'          # use weights pretrained on ImageNet dataset
)

# Freeze base model layers (don't retrain them, saves a LOT of time)
base_model.trainable = False

# Add our own layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)   # converts feature maps to a single vector
x = Dense(128, activation='relu')(x)  # 128 neuron layer, relu = ignore negatives
x = Dropout(0.3)(x)               # randomly turn off 30% neurons (prevents overfitting)
output = Dense(NUM_CLASSES, activation='softmax')(x)  # final layer: 5 class probabilities

# Combine base + our layers into one model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model (set optimizer, loss function, metrics)
model.compile(
    optimizer='adam',                       # adam = best general purpose optimizer
    loss='categorical_crossentropy',        # good for multi-class classification
    metrics=['accuracy']
)

print("    Model built successfully!")
print(f"    Total layers: {len(model.layers)}")


# ============================================================
# STEP 5: Train the Model with MLflow Tracking
# ============================================================
print("\n[3/6] Starting training with MLflow tracking...")

# Create model/ folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Start MLflow run (tracks all params, metrics, and model)
with mlflow.start_run(run_name="MobileNetV2_Flowers"):

    # Log our settings/parameters to MLflow
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("image_size", IMAGE_SIZE)
    mlflow.log_param("model", "MobileNetV2")
    mlflow.log_param("dataset", "Flowers-5class")

    # TRAIN THE MODEL
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        verbose=1   # shows progress bar during training
    )

    # Get final accuracy values
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc   = history.history['val_accuracy'][-1]
    final_val_loss  = history.history['val_loss'][-1]

    print(f"\n    Final Train Accuracy : {final_train_acc:.4f}")
    print(f"    Final Val Accuracy   : {final_val_acc:.4f}")

    # Log final metrics to MLflow
    mlflow.log_metric("final_train_accuracy", final_train_acc)
    mlflow.log_metric("final_val_accuracy",   final_val_acc)
    mlflow.log_metric("final_val_loss",       final_val_loss)


    # ============================================================
    # STEP 6: Plot Accuracy & Loss Graphs
    # ============================================================
    print("\n[4/6] Saving accuracy and loss graphs...")

    epochs_range = range(1, EPOCHS + 1)

    # --- Accuracy Graph ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history['accuracy'],     label='Train Accuracy',      color='blue')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/accuracy_curve.png")   # saves image to model folder
    plt.show()
    print("    Saved: model/accuracy_curve.png")

    # --- Loss Graph ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history['loss'],     label='Train Loss',      color='blue')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("model/loss_curve.png")
    plt.show()
    print("    Saved: model/loss_curve.png")

    # Log graphs to MLflow
    mlflow.log_artifact("model/accuracy_curve.png")
    mlflow.log_artifact("model/loss_curve.png")


    # ============================================================
    # STEP 7: Confusion Matrix
    # ============================================================
    print("\n[5/6] Generating confusion matrix...")

    # Get model predictions on validation data
    val_data.reset()   # reset to start from beginning
    y_pred_probs = model.predict(val_data, verbose=0)   # predicted probabilities
    y_pred = np.argmax(y_pred_probs, axis=1)            # get class index with highest probability
    y_true = val_data.classes                            # actual true labels

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,          # show numbers in each cell
        fmt='d',             # show as integers
        cmap='Blues',        # blue color theme
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png")
    plt.show()
    print("    Saved: model/confusion_matrix.png")

    # Print detailed classification report
    print("\n    Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Log confusion matrix to MLflow
    mlflow.log_artifact("model/confusion_matrix.png")


    # ============================================================
    # STEP 8: Save the Trained Model
    # ============================================================
    print("\n[6/6] Saving trained model...")

    model.save("model/flower_classifier.h5")
    print("    Model saved to: model/flower_classifier.h5")

    # Also log the model to MLflow registry
    mlflow.keras.log_model(model, "flower_model")
    print("    Model logged to MLflow!")


# ============================================================
# DONE!
# ============================================================
print("\n" + "="*50)
print("  TRAINING COMPLETE!")
print("="*50)
print("  Files saved in model/ folder:")
print("    - flower_classifier.h5  (trained model)")
print("    - accuracy_curve.png    (accuracy graph)")
print("    - loss_curve.png        (loss graph)")
print("    - confusion_matrix.png  (confusion matrix)")
print("\n  To view MLflow dashboard, run:")
print("    mlflow ui")
print("  Then open: http://127.0.0.1:5000")
print("="*50)