"""
Handwritten Digit Recognizer
============================
Compares four classic machine learning models on scikit-learn's built-in
"digits" dataset (1,797 small 8x8 grayscale images of handwritten digits 0-9).

What this script does:
  1. Loads the dataset and splits it into training (80%) and test (20%) data.
  2. Trains four different classifiers.
  3. Scores each one with 5-fold cross-validation on the training data.
  4. Picks the best model using the cross-validation score, then evaluates
     it ONCE on the untouched test data.
  5. Saves charts and a text report to the "results/" folder.

Run it with:   python digit_recognizer.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lets the script save charts without opening a window
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42  # fixed seed so results are the same every run
RESULTS_DIR = Path("results")


def load_data():
    """Load the digits dataset and split it into train and test sets."""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=digits.target,  # keep the digit mix equal in both splits
    )
    return digits, X_train, X_test, y_train, y_test


def build_models():
    """Return the models we want to compare.

    Logistic Regression, KNN and SVM work better when features are scaled,
    so they are wrapped in a pipeline with StandardScaler. Random Forest
    does not need scaling.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC()),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }


def compare_models(models, X_train, X_test, y_train, y_test):
    """Train every model and collect its cross-validation and test scores."""
    rows = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        rows.append(
            {
                "name": name,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_acc": test_acc,
            }
        )
        print(
            f"{name:<24} CV accuracy: {cv_scores.mean():.3f} "
            f"(+/- {cv_scores.std():.3f})   Test accuracy: {test_acc:.3f}"
        )
    return rows


def plot_model_comparison(rows):
    """Save a bar chart comparing test accuracy of all models."""
    names = [r["name"] for r in rows]
    accs = [r["test_acc"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, accs, color="#4C72B0")
    ax.set_ylim(90, 100)
    ax.set_ylabel("Test accuracy (%) - note: axis starts at 90%")
    ax.set_title("Model comparison on the digits test set")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + 0.15,
            f"{acc:.1f}%",
            ha="center",
        )
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """Save a confusion matrix: which digits get mixed up with which."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_sample_predictions(model, X_test, y_test):
    """Save a grid of test images with predicted vs. true labels."""
    predictions = model.predict(X_test)
    fig, axes = plt.subplots(3, 6, figsize=(9, 5))
    for ax, image, true, pred in zip(
        axes.ravel(), X_test, y_test, predictions
    ):
        ax.imshow(image.reshape(8, 8), cmap="gray_r")
        color = "green" if true == pred else "red"
        ax.set_title(f"pred {pred} / true {true}", color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample predictions (green = correct, red = wrong)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_predictions.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    digits, X_train, X_test, y_train, y_test = load_data()
    print(
        f"Dataset: {len(digits.data)} images | "
        f"train: {len(X_train)} | test: {len(X_test)}\n"
    )

    models = build_models()
    rows = compare_models(models, X_train, X_test, y_train, y_test)
    plot_model_comparison(rows)

    # Choose the best model using cross-validation (NOT the test set),
    # so the test set stays an honest, unseen check.
    best = max(rows, key=lambda r: r["cv_mean"])
    best_model = models[best["name"]]
    print(f"\nBest model by cross-validation: {best['name']}")

    predictions = best_model.predict(X_test)
    report = classification_report(y_test, predictions)
    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Model: {best['name']}\n"
        f"Test accuracy: {best['test_acc']:.4f}\n\n{report}"
    )
    print("\nClassification report (test set):\n")
    print(report)

    plot_confusion_matrix(best_model, X_test, y_test, best["name"])
    plot_sample_predictions(best_model, X_test, y_test)
    print(f"Charts and report saved in the '{RESULTS_DIR}/' folder.")


if __name__ == "__main__":
    main()
