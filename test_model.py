
import pandas as pd
import joblib

from scipy.sparse import hstack
from feature_extraction import extract_features


# 1. Load saved model components
print("Loading trained model...")

model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
columns = joblib.load("models/feature_columns.pkl")


# 2. Get URL from user
url = input("\nEnter a URL to check: ").strip()

if not url:
    print("Error: Please enter a URL.")
    raise SystemExit


# 3. Extract numerical features
numeric_features = pd.DataFrame(
    [extract_features(url)]
)[columns]


# 4. Extract TF-IDF text features
text_features = vectorizer.transform([url])


# 5. Combine both feature types
combined_features = hstack([
    numeric_features.to_numpy(dtype="float32"),
    text_features
]).tocsr()


# 6. Predict URL category
prediction = model.predict(combined_features)[0]
probabilities = model.predict_proba(combined_features)[0]

confidence = max(probabilities) * 100


# 7. Display result
print("\n========== URL ANALYSIS ==========")
print("URL:", url)

if prediction == 0:
    print("Result: PHISHING")
else:
    print("Result: LEGITIMATE")

print("Model confidence:", round(confidence, 2), "%")
print("==================================")
