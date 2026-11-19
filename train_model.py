
import pandas as pd
import numpy as np
import joblib

from urllib.parse import urlparse
from scipy.sparse import hstack

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. Load datasets
print("Loading datasets...")

df = pd.read_csv("dataset/unique_phishing_dataset.csv")
features = pd.read_csv("dataset/url_features_dataset.csv")

urls = df["URL"].astype(str)
X_numeric = features.drop(columns=["label"])
y = df["label"]


# 2. Extract hostname from each URL
print("\nExtracting hostnames...")

groups = urls.apply(
    lambda url: urlparse(
        url if "://" in url else "http://" + url
    ).hostname
)

groups = groups.fillna("")


# 3. Verify feature and URL alignment
if not features["label"].equals(df["label"]):
    raise ValueError("Feature labels do not match URL labels!")

print("Dataset alignment verified.")


# 4. Split by hostname
print("\nSplitting dataset by hostname...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X_numeric, y, groups)
)

X_train_num = X_numeric.iloc[train_idx]
X_test_num = X_numeric.iloc[test_idx]

url_train = urls.iloc[train_idx]
url_test = urls.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))

print(
    "Hostname overlap:",
    len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
)


# 5. Extract TF-IDF features
print("\nExtracting TF-IDF features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=50000,
    min_df=2,
    dtype=np.float32
)

X_train_tfidf = vectorizer.fit_transform(url_train)
X_test_tfidf = vectorizer.transform(url_test)


# 6. Combine numerical and text features
print("\nCombining features...")

X_train = hstack([
    X_train_num.to_numpy(dtype=np.float32),
    X_train_tfidf
]).tocsr()

X_test = hstack([
    X_test_num.to_numpy(dtype=np.float32),
    X_test_tfidf
]).tocsr()

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)


# 7. Train Logistic Regression
print("\nTraining model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    solver="liblinear"
)

model.fit(X_train, y_train)

print("Training completed!")


# 8. Evaluate model
print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nDomain-based accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# 9. Save trained components
print("\nSaving trained model...")

joblib.dump(model, "models/phishing_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(list(X_numeric.columns), "models/feature_columns.pkl")

print("\nAll model files saved successfully!")

