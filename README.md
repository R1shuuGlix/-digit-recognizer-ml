# Handwritten Digit Recognizer

A beginner-friendly machine learning project that compares four classic classification models on handwritten digit images, and reports which one performs best.

**Tech:** Python, scikit-learn, Matplotlib

## What it does

1. Loads the **digits dataset** bundled with scikit-learn (1,797 grayscale images, 8x8 pixels each, digits 0-9; derived from the UCI Optical Recognition of Handwritten Digits dataset).
2. Splits the data into **80% training / 20% testing** (1,437 train, 360 test), keeping the digit mix equal in both sets.
3. Trains four models:
   - Logistic Regression
   - K-Nearest Neighbors (k = 5)
   - Support Vector Machine (SVM)
   - Random Forest (200 trees)
4. Scores each model with **5-fold cross-validation** on the training data.
5. Picks the best model using the cross-validation score (not the test set, so the test set stays an honest, unseen check), then evaluates it once on the test set.
6. Saves charts and a classification report to the `results/` folder.

## Results

| Model                  | 5-fold CV accuracy | Test accuracy |
| ---------------------- | ------------------ | ------------- |
| Logistic Regression    | 97.0%              | 97.2%         |
| K-Nearest Neighbors    | 97.5%              | 96.4%         |
| **Support Vector Machine** | **98.3%**      | **97.5%**     |
| Random Forest          | 97.5%              | 96.4%         |

The **SVM** had the best cross-validation score, and scored **97.5% (351 of 360)** on the unseen test images. The four models are close to each other, so the differences are small. With only 360 test images, one or two extra mistakes can change a score noticeably. Exact numbers may differ slightly across library versions.

### Model comparison
![Model comparison](results/model_comparison.png)

### Confusion matrix (best model)
Shows which digits get mixed up with which. Rows are the true digit, columns are the predicted digit.

![Confusion matrix](results/confusion_matrix.png)

### Sample predictions
![Sample predictions](results/sample_predictions.png)

The full per-digit report (precision, recall, F1) is in [`results/classification_report.txt`](results/classification_report.txt).

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/R1shuuGlix/digit-recognizer-ml.git
cd digit-recognizer-ml

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python digit_recognizer.py
```

The script prints each model's scores in the terminal and refreshes the files in `results/`. A fixed random seed (`RANDOM_STATE = 42`) makes runs repeatable.

## Project structure

```
digit-recognizer-ml/
├── digit_recognizer.py     # main script: load data, train, compare, plot
├── requirements.txt        # Python dependencies
├── README.md
└── results/
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── classification_report.txt
```

## Possible improvements

- Tune hyperparameters with `GridSearchCV` (for example the SVM's `C` and `gamma`).
- Try a small neural network (`MLPClassifier`) or a CNN on the larger MNIST dataset.
- Add a script that predicts a digit from the user's own image.
- Save the trained model with `joblib` and load it in a small web app.

## Author

Rishabh Nagar - [LinkedIn](https://www.linkedin.com/in/rishabh-nagar-35361a425/)
