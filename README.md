# CI/CD Pipeline Monitoring using CNN + OpenCV

A deep learning system that looks at a screenshot of a GitHub Actions
dashboard and predicts whether the pipeline run was a **SUCCESS** or a
**FAILURE**, using a traditional CNN (no transfer learning) with OpenCV
preprocessing.

---

## 1. Project Structure

```
cicd_pipeline_monitor/
├── dataset/
│   ├── success/        <- put your "success" screenshots here
│   └── failure/        <- put your "failure" screenshots here
├── processed/           <- optional debug output from preprocess.py
├── src/
│   ├── preprocess.py     <- OpenCV preprocessing pipeline (shared by train + predict)
│   ├── train.py          <- loads data, builds CNN, trains, evaluates, saves model
│   └── predict.py        <- loads saved model and predicts a single new image
├── models/               <- trained model (cicd_pipeline_monitor.h5) saved here
├── reports/              <- accuracy/loss plots, confusion matrix, classification report
├── requirements.txt
└── README.md
```

---

## 2. Setup (do this first)

1. Install Python 3.9–3.11 (TensorFlow does not yet support every newer
   Python version, so staying in this range avoids install headaches).
2. Open a terminal inside the `cicd_pipeline_monitor/` folder.
3. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Add your dataset (you have not done this yet — required before training)

Right now `dataset/success/` and `dataset/failure/` are **empty**. Training
will not run until you add images.

- Put "pipeline succeeded" screenshots into `dataset/success/`
- Put "pipeline failed" screenshots into `dataset/failure/`
- Any common image format works (`.png`, `.jpg`, `.jpeg`) — naming like
  `success_1.png`, `success_2.png`, etc. is just a convention, not a
  requirement.
- You do **not** need exactly 10,000 images to get started. Even 30–50
  images per class is enough to confirm the entire pipeline runs end to
  end. You can add more images later and re-run training — more data
  almost always improves accuracy.

If you have no real screenshots yet, you can take a few yourself: open any
public GitHub Actions run (green checkmark = success, red X = failure) and
save a screenshot of the dashboard for each class.

---

## 4. Test preprocessing in isolation (optional but recommended)

Before training, you can sanity-check the OpenCV pipeline on a single image:

```bash
cd src
python preprocess.py
```

This reads one sample image, runs it through the full pipeline (resize,
BGR→RGB, crop, blur, normalize), prints its shape and pixel value range,
and saves a debug copy to `processed/debug_sample.png` so you can visually
confirm nothing looks broken (e.g., colors aren't swapped, important
content wasn't cropped off).

---

## 5. Train the model

Once you have images in both `dataset/success/` and `dataset/failure/`:

```bash
cd src
python train.py
```

This will, in order:
1. Load and preprocess every image in both folders
2. Split into 80% training / 20% testing (stratified, so class balance is preserved)
3. Build the 3-block CNN (Conv2D → BatchNorm → MaxPool → Dropout, ×3, then a Dense head)
4. Apply data augmentation (rotation, zoom, width/height shift) to training data only
5. Train for 20 epochs
6. Evaluate on the test set (accuracy, classification report, confusion matrix)
7. Save plots of accuracy/loss curves and the confusion matrix into `reports/`
8. Save the trained model to `models/cicd_pipeline_monitor.h5`

If your dataset folders are empty, `train.py` will stop immediately with a
clear error message telling you exactly which folders need images, instead
of crashing with a confusing stack trace.

---

## 6. Predict on a new screenshot

Once a model has been trained and saved:

```bash
cd src
python predict.py ../test.png
```

(or just `python predict.py` if you place a screenshot at
`cicd_pipeline_monitor/test.png`)

Output looks like:

```
Pipeline Status : SUCCESS
```

or

```
Pipeline Status : FAILURE
```

---

## 7. Notes on scaling up the dataset later

The project spec targets 5,000 success + 5,000 failure images. Realistically,
there's no existing public dataset of GitHub Actions screenshots, so you'll
need to either:
- Generate synthetic dashboard screenshots programmatically, or
- Collect/screenshot real CI runs over time, or
- Start small (tens of images) and grow the dataset gradually.

The code does not need to change as your dataset grows — just keep adding
images into `dataset/success/` and `dataset/failure/` and re-run `train.py`.
