import pandas as pd
import joblib

from urllib.parse import urlparse
from flask import Flask, render_template, request
from scipy.sparse import hstack

from feature_extraction import extract_features

app = Flask(__name__)

# Load trained model components
model = joblib.load("models/phishing_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
columns = joblib.load("models/feature_columns.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    url = ""
    error = None

    if request.method == "POST":

        url = request.form.get("url", "").strip()

        # Validate URL
        parsed_url = urlparse(url)

        if not url:
            error = "Please enter a website URL."

        elif parsed_url.scheme not in ("http", "https"):
            error = "URL must start with http:// or https://."

        elif not parsed_url.hostname:
            error = "Please enter a valid website address."

        else:
            # Extract numerical features
            numeric_features = pd.DataFrame(
                [extract_features(url)]
            )[columns]

            # Extract TF-IDF features
            text_features = vectorizer.transform([url])

            # Combine features
            combined_features = hstack([
                numeric_features.to_numpy(dtype="float32"),
                text_features
            ]).tocsr()

            # Predict URL category
            prediction = model.predict(combined_features)[0]

            probabilities = model.predict_proba(
                combined_features
            )[0]

            confidence = round(max(probabilities) * 100, 2)

            if prediction == 0:
                result = "PHISHING"
            else:
                result = "LEGITIMATE"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        url=url,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
