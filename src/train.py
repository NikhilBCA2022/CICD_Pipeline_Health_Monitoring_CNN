# =====================================================
# IMPORTS
# =====================================================
# OS and glob help us walk through the dataset folders and find image files.
import os
import glob

# NumPy is used to build the X (images) and y (labels) arrays the CNN needs.
import numpy as np

# Matplotlib is used to plot training/validation accuracy and loss curves.
import matplotlib.pyplot as plt

# Scikit-learn gives us the train/test split and evaluation metrics.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# TensorFlow / Keras gives us the CNN building blocks, the optimizer, and the
# image augmentation generator.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# We import our OWN preprocessing function so that training uses the EXACT
# same image pipeline that predict.py will use later. This is critical —
# if these two ever drift apart, the model will perform poorly in production.
from preprocess import preprocess_image, IMAGE_SIZE


# =====================================================
# CONFIGURATION
# =====================================================
# Paths are relative to this file's location (src/), so this script works
# correctly no matter where the user launches it from, as long as they run
# it from inside src/.
DATASET_DIR = "../dataset"
SUCCESS_DIR = os.path.join(DATASET_DIR, "success")
FAILURE_DIR = os.path.join(DATASET_DIR, "failure")

MODELS_DIR = "../models"
REPORTS_DIR = "../reports"

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "cicd_pipeline_monitor.h5")

# Labels: 0 = SUCCESS, 1 = FAILURE (matches the project specification).
LABEL_SUCCESS = 0
LABEL_FAILURE = 1

EPOCHS = 20
BATCH_SIZE = 32
TEST_SPLIT = 0.20   # 20% test, 80% train
RANDOM_SEED = 42    # fixed seed so results are reproducible across runs

# Make sure output folders exist before we try to save anything into them.
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================
def load_dataset():
    """
    Walks through dataset/success and dataset/failure, preprocesses every
    image using the SAME function predict.py will use, and builds the final
    X (images) and y (labels) arrays for training.

    WHY a dedicated function: keeping data loading separate from model
    building/training makes the script easier to read and easier to debug
    (you can test loading in isolation before worrying about the CNN).
    """
    image_paths = []
    labels = []

    # Collect every success image path with label 0.
    success_paths = sorted(glob.glob(os.path.join(SUCCESS_DIR, "*")))
    image_paths.extend(success_paths)
    labels.extend([LABEL_SUCCESS] * len(success_paths))

    # Collect every failure image path with label 1.
    failure_paths = sorted(glob.glob(os.path.join(FAILURE_DIR, "*")))
    image_paths.extend(failure_paths)
    labels.extend([LABEL_FAILURE] * len(failure_paths))

    # ---------------------------------------------------------------
    # GUARDRAIL: stop with a clear, friendly message if there is no data.
    # Without this check, the script would fail later with a confusing
    # NumPy/TensorFlow error that doesn't explain the real problem.
    # ---------------------------------------------------------------
    if len(image_paths) == 0:
        raise RuntimeError(
            "\n\nNo images found!\n"
            f"Looked in:\n  {os.path.abspath(SUCCESS_DIR)}\n  {os.path.abspath(FAILURE_DIR)}\n\n"
            "Add screenshot images into dataset/success/ and dataset/failure/ "
            "before running train.py.\n"
        )

    print(f"Found {len(success_paths)} success images and {len(failure_paths)} failure images.")
    print(f"Total images to process: {len(image_paths)}")

    # Preprocess every image one by one using our shared preprocessing
    # pipeline (resize, color convert, crop, blur, normalize).
    images = []
    valid_labels = []

    for path, label in zip(image_paths, labels):
        try:
            processed = preprocess_image(path)
            images.append(processed)
            valid_labels.append(label)
        except Exception as error:
            # If a single corrupted image fails, we skip it instead of
            # crashing the whole training run, but we tell the user so
            # they can investigate if many images are failing.
            print(f"Skipping {path} due to error: {error}")

    X = np.array(images, dtype=np.float32)
    y = np.array(valid_labels, dtype=np.float32)

    print(f"Successfully loaded {X.shape[0]} images with shape {X.shape[1:]}")

    return X, y


# =====================================================
# BUILD CNN
# =====================================================
def build_cnn_model(input_shape=(224, 224, 3)):
    """
    Builds a traditional Sequential CNN (no transfer learning), exactly
    following the 3-block architecture specification:

        BLOCK 1: Conv2D(32)  -> BatchNorm -> MaxPool -> Dropout
        BLOCK 2: Conv2D(64)  -> BatchNorm -> MaxPool -> Dropout
        BLOCK 3: Conv2D(128) -> BatchNorm -> MaxPool -> Dropout
        Flatten -> Dense(256) -> BatchNorm -> Dropout -> Dense(1, sigmoid)

    WHY each piece:
        - Conv2D layers learn visual patterns (edges, shapes, colors) at
          increasing complexity as we go deeper (32 -> 64 -> 128 filters).
        - BatchNormalization stabilizes and speeds up training by keeping
          layer outputs in a consistent numeric range.
        - MaxPooling reduces the spatial size of feature maps, which both
          speeds up computation and helps the model focus on the strongest
          features rather than exact pixel positions.
        - Dropout randomly disables neurons during training to prevent
          overfitting (the model memorizing the training set instead of
          learning general patterns).
        - L2 regularization adds a small penalty for large weights, which
          further discourages overfitting.
        - The final Dense(1) + sigmoid outputs a single probability between
          0 and 1, perfect for our binary classification (success/failure).
    """
    # L2 regularization strength. A small value (0.001) gently discourages
    # overly large weights without overpowering the main loss signal.
    l2_strength = 0.001

    model = Sequential()

    # -------------------- BLOCK 1 --------------------
    # First convolution block: learns simple, low-level features like
    # edges and color blobs (e.g., the edges of a green/red status banner).
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_strength),
        input_shape=input_shape
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # -------------------- BLOCK 2 --------------------
    # Second convolution block: learns more complex shapes by combining
    # the simple features from Block 1 (e.g., checkmark/cross icon shapes).
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_strength)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # -------------------- BLOCK 3 --------------------
    # Third convolution block: learns high-level, abstract patterns that
    # combine shapes and colors into a holistic sense of "this dashboard
    # looks like a failure" or "this dashboard looks like a success".
    model.add(Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(l2_strength)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # -------------------- CLASSIFIER HEAD --------------------
    # Flatten converts the 3D feature maps into a 1D vector so the Dense
    # (fully connected) layers can process them.
    model.add(Flatten())

    # Dense(256): a fully connected layer that combines all extracted
    # features to learn the final decision boundary between the two classes.
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(l2_strength)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # stronger dropout here since Dense layers overfit easily

    # Final output layer: 1 neuron with sigmoid activation outputs a value
    # between 0 and 1. We interpret values close to 0 as SUCCESS and values
    # close to 1 as FAILURE (matching our label convention).
    model.add(Dense(1, activation="sigmoid"))

    # -------------------- COMPILE --------------------
    # Adam optimizer: an adaptive learning-rate optimizer that generally
    # trains faster and more reliably than plain SGD for image tasks.
    optimizer = Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",   # standard loss for binary classification
        metrics=["accuracy"]
    )

    # Print a full summary so the user can see every layer, its output
    # shape, and its parameter count.
    model.summary()

    return model


# =====================================================
# TRAIN MODEL
# =====================================================
def train_model(model, X_train, y_train, X_test, y_test):
    """
    Trains the CNN using augmented training data and validates against the
    held-out test set after every epoch.

    WHY ImageDataGenerator: screenshots in the real world will never be
    perfectly centered or zoomed identically. By randomly rotating, zooming,
    and shifting training images, we teach the model to be robust to small
    visual variations instead of memorizing exact pixel positions.
    """
    # Data augmentation is applied ONLY to training data. We never augment
    # test data, because we want test metrics to reflect performance on
    # realistic, unmodified images.
    train_augmentor = ImageDataGenerator(
        rotation_range=10,       # small rotations, since screenshots are rarely tilted heavily
        zoom_range=0.1,          # slight zoom in/out
        width_shift_range=0.1,   # slight horizontal shifting
        height_shift_range=0.1   # slight vertical shifting
    )

    train_generator = train_augmentor.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE
    )

    # Fit the model. We pass the test set as validation_data so we can
    # monitor generalization performance after every epoch.
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )

    return history


# =====================================================
# EVALUATION
# =====================================================
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and prints/saves:
        - Overall accuracy
        - Classification report (precision, recall, f1-score per class)
        - Confusion matrix (as both numbers and a saved heatmap image)
    """
    # Get raw probability predictions (values between 0 and 1).
    y_pred_probs = model.predict(X_test)

    # Convert probabilities into hard class predictions using a 0.5 threshold.
    # Above 0.5 -> FAILURE (1), below 0.5 -> SUCCESS (0).
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # ---------------- Accuracy ----------------
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # ---------------- Classification Report ----------------
    report = classification_report(
        y_test, y_pred,
        target_names=["SUCCESS", "FAILURE"]
    )
    print("\nClassification Report:\n")
    print(report)

    # Save the report to a text file so it's easy to reference later.
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Plot the confusion matrix as a heatmap for easy visual interpretation.
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["SUCCESS", "FAILURE"],
        yticklabels=["SUCCESS", "FAILURE"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    confusion_matrix_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_matrix_path}")

    return accuracy


# =====================================================
# VISUALIZATION
# =====================================================
def plot_training_history(history):
    """
    Plots training/validation accuracy and training/validation loss curves
    side by side and saves the figure to reports/training_history.png.

    WHY: these curves are the fastest way to visually diagnose problems:
        - If training accuracy keeps rising but validation accuracy flattens
          or drops, the model is overfitting.
        - If both losses stay high, the model is underfitting or the
          learning rate may need adjustment.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------- Accuracy Plot ----------------
    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # ---------------- Loss Plot ----------------
    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    history_plot_path = os.path.join(REPORTS_DIR, "training_history.png")
    plt.savefig(history_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to {history_plot_path}")


# =====================================================
# SAVE MODEL
# =====================================================
def save_model(model):
    """
    Saves the trained model to disk in HDF5 (.h5) format so it can be
    loaded later by predict.py without retraining.
    """
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {os.path.abspath(MODEL_SAVE_PATH)}")


# =====================================================
# MAIN ENTRY POINT
# =====================================================
def main():
    print("=" * 55)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 55)
    X, y = load_dataset()

    print("\n" + "=" * 55)
    print("STEP 2: SPLITTING DATA (80% TRAIN / 20% TEST)")
    print("=" * 55)
    # stratify=y ensures both train and test sets keep the same success/failure
    # ratio as the full dataset, which is important for a fair evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"Training images: {X_train.shape[0]}")
    print(f"Testing images:  {X_test.shape[0]}")

    print("\n" + "=" * 55)
    print("STEP 3: BUILDING CNN MODEL")
    print("=" * 55)
    model = build_cnn_model(input_shape=X_train.shape[1:])

    print("\n" + "=" * 55)
    print("STEP 4: TRAINING MODEL")
    print("=" * 55)
    history = train_model(model, X_train, y_train, X_test, y_test)

    print("\n" + "=" * 55)
    print("STEP 5: EVALUATING MODEL")
    print("=" * 55)
    evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 55)
    print("STEP 6: PLOTTING TRAINING HISTORY")
    print("=" * 55)
    plot_training_history(history)

    print("\n" + "=" * 55)
    print("STEP 7: SAVING MODEL")
    print("=" * 55)
    save_model(model)

    print("\nAll done! You can now run predict.py to test the model on a new image.")


if __name__ == "__main__":
    main()
