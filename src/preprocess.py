# =====================================================
# IMPORTS
# =====================================================
# We use OpenCV (cv2) for all image reading and image processing operations.
# NumPy is used because OpenCV images are just NumPy arrays under the hood,
# and we need NumPy to normalize pixel values for the CNN.
import cv2
import numpy as np
import os


# =====================================================
# CONFIGURATION
# =====================================================
# The CNN expects a fixed input size. 224x224 is a common, CNN-friendly size
# that keeps enough detail (text, icons, colors) from a dashboard screenshot
# without making the model too slow to train.
IMAGE_SIZE = (224, 224)

# When we crop a screenshot, we remove a percentage of pixels from each side.
# This is useful because GitHub Actions dashboard screenshots often contain
# browser chrome, sidebars, or repo navigation bars that do NOT help the model
# decide "success" vs "failure". Cropping them out keeps the model focused on
# the actual pipeline status area.
CROP_PERCENT = 0.05  # crop 5% from each side


# =====================================================
# STEP 1: READ IMAGE
# =====================================================
def read_image(image_path):
    """
    Reads an image from disk using OpenCV.

    WHY: cv2.imread() is the standard way to load an image into a NumPy
    array so we can manipulate it with OpenCV functions. OpenCV loads
    images in BGR format by default (not RGB), which is why we will need
    to convert it later.
    """
    image = cv2.imread(image_path)

    # If the path is wrong or the file is corrupted, cv2.imread returns None.
    # We check for this immediately so we get a clear error instead of a
    # confusing crash later in the pipeline.
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    return image


# =====================================================
# STEP 2: RESIZE IMAGE
# =====================================================
def resize_image(image, size=IMAGE_SIZE):
    """
    Resizes the image to a fixed size (224x224).

    WHY: CNNs require every input image to have the SAME dimensions because
    the network's weights are built for a fixed input shape. Screenshots can
    come in many different resolutions, so we standardize them here.
    """
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized


# =====================================================
# STEP 3: CONVERT BGR -> RGB
# =====================================================
def convert_bgr_to_rgb(image):
    """
    Converts an image from BGR color order to RGB color order.

    WHY: OpenCV reads images in BGR order (Blue, Green, Red) for historical
    reasons, but almost every deep learning framework (TensorFlow, Keras,
    Matplotlib) expects RGB order. If we skip this step, the colors will be
    swapped and the model may learn incorrect color-based patterns
    (for example, red "failure" badges could look blue).
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


# =====================================================
# STEP 4: CROP UNNECESSARY DASHBOARD BORDERS
# =====================================================
def crop_borders(image, crop_percent=CROP_PERCENT):
    """
    Crops a fixed percentage of pixels from each edge of the image.

    WHY: Dashboard screenshots often include browser tabs, scrollbars, or
    sidebar navigation that is irrelevant to predicting success/failure.
    Cropping the outer border removes this noise and forces the model to
    focus on the central pipeline status content.
    """
    height, width = image.shape[:2]

    # Calculate how many pixels to remove from top/bottom and left/right.
    crop_h = int(height * crop_percent)
    crop_w = int(width * crop_percent)

    # Slice the image array to remove the border pixels.
    cropped = image[crop_h:height - crop_h, crop_w:width - crop_w]

    return cropped


# =====================================================
# STEP 5: GAUSSIAN BLUR
# =====================================================
def apply_gaussian_blur(image, kernel_size=(3, 3)):
    """
    Applies a light Gaussian blur to the image.

    WHY: Screenshots can contain compression artifacts, sharp text edges,
    and small UI noise (anti-aliasing pixels) that can distract the CNN
    during training. A light blur smooths out this high-frequency noise
    while still preserving the overall shapes and colors the CNN needs
    (e.g., green success banners vs red failure banners).
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigmaX=0)
    return blurred


# =====================================================
# STEP 6: OPTIONAL GRAYSCALE EXPERIMENT
# =====================================================
def convert_to_grayscale(image):
    """
    Converts the image to grayscale.

    WHY: This is an OPTIONAL experiment. Sometimes color is not actually
    necessary to detect success/failure (e.g., if the model can rely on
    icon shapes like checkmarks vs crosses, or text patterns). Testing a
    grayscale version helps us check whether color is truly adding value,
    or whether the model can perform just as well with less information
    (which would make the model faster and simpler).

    NOTE: This function is provided separately and is NOT used in the main
    pipeline by default, since our CNN is built to accept 3-channel RGB
    images. Use it only if you want to manually experiment with a
    grayscale-based model variant.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


# =====================================================
# STEP 7: NORMALIZE PIXELS
# =====================================================
def normalize_image(image):
    """
    Normalizes pixel values from the [0, 255] range to the [0, 1] range.

    WHY: Neural networks train faster and more stably when input values are
    small and centered in a consistent range. Raw pixel values (0-255) can
    cause large gradients and slow, unstable training. Dividing by 255.0
    converts every pixel into a float between 0 and 1.
    """
    # Convert to float32 first so division produces decimal values, not
    # integer division (which would just give us 0s and 1s incorrectly).
    normalized = image.astype(np.float32) / 255.0
    return normalized


# =====================================================
# STEP 8: SAVE PROCESSED OUTPUT
# =====================================================
def save_processed_image(image, save_path):
    """
    Saves a processed image to disk.

    WHY: Saving intermediate processed images lets us visually inspect what
    the CNN will actually "see" during training. This is useful for
    debugging (e.g., checking that crop_borders didn't accidentally cut off
    important content).

    NOTE: Since our pipeline normalizes pixels to [0, 1] floats, we must
    convert back to [0, 255] integers before saving, otherwise the saved
    image will appear almost completely black.
    """
    # Undo normalization for saving purposes only (training data stays normalized).
    image_to_save = (image * 255.0).astype(np.uint8)

    # cv2.imwrite expects BGR order, so we convert back from RGB to BGR
    # before writing the file, otherwise saved images will have swapped colors.
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image_to_save)


# =====================================================
# FULL PREPROCESSING PIPELINE (USED BY BOTH TRAIN + PREDICT)
# =====================================================
def preprocess_image(image_path, save_debug_path=None):
    """
    Runs the FULL preprocessing pipeline on a single image, step by step.

    WHY THIS FUNCTION EXISTS: Both train.py and predict.py need to process
    images in EXACTLY the same way. If training preprocessing and prediction
    preprocessing ever drift apart (for example, one resizes differently),
    the model will perform poorly at prediction time. By putting all steps
    in ONE shared function, we guarantee consistency.

    Steps performed (in order):
        1. Read image
        2. Resize to 224x224
        3. Convert BGR -> RGB
        4. Crop unnecessary dashboard borders
        5. Gaussian blur
        6. Normalize pixels to [0, 1]

    Returns:
        A normalized NumPy array of shape (224, 224, 3), ready for the CNN.
    """
    # Step 1: Read image from disk.
    image = read_image(image_path)

    # Step 2: Resize to the fixed CNN input size.
    image = resize_image(image)

    # Step 3: Convert color order from BGR to RGB.
    image = convert_bgr_to_rgb(image)

    # Step 4: Crop irrelevant dashboard borders.
    image = crop_borders(image)

    # Since cropping changes the image size, we resize again to lock the
    # final shape back to 224x224 before feeding it to the CNN.
    image = resize_image(image)

    # Step 5: Apply a light Gaussian blur to reduce noise.
    image = apply_gaussian_blur(image)

    # Step 6: Normalize pixel values to [0, 1] for stable CNN training.
    image = normalize_image(image)

    # Optional debug save: write out the processed image so we can visually
    # confirm the pipeline is working as expected.
    if save_debug_path is not None:
        save_processed_image(image, save_debug_path)

    return image


# =====================================================
# QUICK MANUAL TEST (RUN THIS FILE DIRECTLY TO TEST)
# =====================================================
if __name__ == "__main__":
    # This block only runs if you execute "python preprocess.py" directly.
    # It is a simple sanity check, not part of the main training pipeline.
    sample_path = "../dataset/success/success_0001.png"

    if os.path.exists(sample_path):
        processed = preprocess_image(sample_path, save_debug_path="../processed/debug_sample.png")
        print("Processed image shape:", processed.shape)
        print("Pixel value range: min =", processed.min(), "max =", processed.max())
    else:
        print(f"Sample image not found at {sample_path}. "
              f"Add images to dataset/success/ to test preprocessing.")
