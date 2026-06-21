# =====================================================
# IMPORTS
# =====================================================
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

# Same shared preprocessing function used during training. This is the
# most important import in this file — using anything else here would
# break the model's predictions.
from preprocess import preprocess_image


# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "../models/cicd_pipeline_monitor.h5"
DEFAULT_TEST_IMAGE = "../test_img.png"

# Same label convention used in train.py: 0 = SUCCESS, 1 = FAILURE.
LABEL_NAMES = {0: "SUCCESS", 1: "FAILURE"}


# =====================================================
# LOAD MODEL
# =====================================================
def load_trained_model(model_path=MODEL_PATH):
    """
    Loads the previously trained and saved CNN model from disk.

    WHY a dedicated function: keeping model loading separate makes it easy
    to reuse this logic if predict.py is ever imported by another script
    (e.g., a future Flask/FastAPI service that monitors pipelines live).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n\nNo trained model found at {os.path.abspath(model_path)}.\n"
            "Run train.py first to train and save the model.\n"
        )

    model = load_model(model_path)
    return model


# =====================================================
# PREDICT SINGLE IMAGE
# =====================================================
def predict_image(model, image_path):
    """
    Runs the full prediction pipeline on a single image:
        1. Preprocess the image (identical to training preprocessing)
        2. Add a batch dimension (the model expects a batch of images,
           even if that batch only contains 1 image)
        3. Run the model's forward pass to get a probability
        4. Convert the probability into a SUCCESS/FAILURE label
    """
    # Step 1: Preprocess exactly like training data.
    processed_image = preprocess_image(image_path)

    # Step 2: Models expect input shape (batch_size, height, width, channels).
    # A single image has shape (224, 224, 3), so we add a new axis at the
    # front to make it (1, 224, 224, 3).
    batched_image = np.expand_dims(processed_image, axis=0)

    # Step 3: Get the raw probability output (a value between 0 and 1).
    probability = model.predict(batched_image)[0][0]

    # Step 4: Apply the same 0.5 threshold used during evaluation.
    # Values below 0.5 -> SUCCESS (0), values above 0.5 -> FAILURE (1).
    predicted_label = 1 if probability > 0.5 else 0

    return predicted_label, probability


# =====================================================
# MAIN ENTRY POINT
# =====================================================
def main():
    # Allow the user to optionally pass a custom image path as a command
    # line argument: python predict.py path/to/image.png
    # If no argument is given, we fall back to the default test.png path.
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = DEFAULT_TEST_IMAGE

    print("=" * 55)
    print("STEP 1: LOADING TRAINED MODEL")
    print("=" * 55)
    model = load_trained_model()

    print("\n" + "=" * 55)
    print("STEP 2: PREPROCESSING TEST IMAGE")
    print("=" * 55)
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"\n\nTest image not found at {os.path.abspath(image_path)}.\n"
            "Place a screenshot at that path, or pass a custom path like:\n"
            "    python predict.py path/to/your_screenshot.png\n"
        )
    print(f"Using image: {os.path.abspath(image_path)}")

    print("\n" + "=" * 55)
    print("STEP 3: RUNNING PREDICTION")
    print("=" * 55)
    predicted_label, probability = predict_image(model, image_path)
    status = LABEL_NAMES[predicted_label]

    print(f"Raw model probability (closer to 1 = FAILURE): {probability:.4f}")
    print(f"\nPipeline Status : {status}")


if __name__ == "__main__":
    main()
