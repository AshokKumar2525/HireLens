#!/usr/bin/env python3
import os
import re
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, redirect, url_for, flash

# -------------------
# Paths & Config
# -------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "model/vectorizer.joblib")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = SECRET_KEY


# -------------------
# Load Model + Vectorizer
# -------------------
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# -------------------
# Meta Feature Extraction
# -------------------
def extract_meta_features(description: str) -> dict:
    desc = description.lower()

    telecommuting = 1 if "remote" in desc or "work from home" in desc else 0
    has_company_logo = 0
    has_questions = 1 if "?" in description else 0

    if "full time" in desc:
        employment_type = "Full-time"
    elif "part time" in desc:
        employment_type = "Part-time"
    elif "contract" in desc:
        employment_type = "Contract"
    else:
        employment_type = "Other"

    if "years" in desc or "experience" in desc:
        required_experience = "Experienced"
    elif "fresher" in desc or "entry level" in desc:
        required_experience = "Entry level"
    else:
        required_experience = "Not Specified"

    if "bachelor" in desc or "degree" in desc:
        required_education = "Bachelor's"
    elif "master" in desc:
        required_education = "Master's"
    elif "phd" in desc:
        required_education = "PhD"
    else:
        required_education = "Not Specified"

    if "software" in desc or "developer" in desc:
        industry = "Software"
    elif "sales" in desc:
        industry = "Sales"
    elif "data" in desc:
        industry = "Data"
    else:
        industry = "Other"

    if "engineer" in desc:
        function = "Engineering"
    elif "manager" in desc:
        function = "Management"
    elif "analyst" in desc:
        function = "Analyst"
    else:
        function = "Other"

    # return categorical values
    return {
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
    }


# -------------------
# Convert Meta Features to Numeric
# -------------------
def encode_meta(meta: dict) -> np.ndarray:
    return np.array([[
        int(meta["telecommuting"]),
        int(meta["has_company_logo"]),
        int(meta["has_questions"]),
        len(meta["employment_type"]),      # crude encoding
        len(meta["required_experience"]),
        len(meta["required_education"]),
        len(meta["industry"]),
        len(meta["function"])
    ]])


# -------------------
# Extract Info (for display)
# -------------------
def extract_fields(text: str):
    title_match = re.search(r"(?:^|\n)\s*(?:title|job title)[:\-]\s*(.+)", text, flags=re.I)
    loc_match = re.search(r"(?:^|\n)\s*(?:location)[:\-]\s*(.+)", text, flags=re.I)
    email_match = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.findall(r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", text)
    urls = re.findall(r"https?://\S+|www\.\S+", text)
    return {
        "title": title_match.group(1).strip() if title_match else "",
        "location": loc_match.group(1).strip() if loc_match else "",
        "emails": list(set(email_match))[:3],
        "phones": list(set(phone_match))[:3],
        "links": list(set(urls))[:5],
        "length": len(text.split()),
    }


# -------------------
# Fetch Job Posting from URL
# -------------------
def fetch_from_url(url: str) -> str:
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text.strip()
    except Exception as e:
        print("URL fetch error:", e)
        return ""


# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction, prob_fake, extracted, text_val = None, None, {}, ""

    if request.method == "POST":
        text_val = request.form.get("job_post", "").strip()
        url_val = request.form.get("job_url", "").strip()

        # Auto-detect URL in text field
        if text_val and re.match(r"^(https?://|www\.)", text_val):
            url_val = text_val
            text_val = ""

        if url_val and not text_val:
            text_val = fetch_from_url(url_val)
            if not text_val:
                flash("Could not extract text from the provided link.", "warning")
                return redirect(url_for("index"))

        if not text_val:
            flash("Please paste a job description or provide a job link.", "warning")
            return redirect(url_for("index"))

        try:
            model, vectorizer = load_artifacts()
        except Exception as e:
            flash(str(e), "error")
            return redirect(url_for("index"))

        extracted = extract_fields(text_val)

        try:
            # TF-IDF features (100)
            tfidf_features = vectorizer.transform([text_val]).toarray()

            # Meta features (8 → numeric)
            meta = extract_meta_features(text_val)
            meta_vector = encode_meta(meta)

            # Combine into final 108 features
            features = np.hstack([tfidf_features, meta_vector])

            # Predict
            pred = model.predict(features)[0]
            prediction = "FAKE Job Posting 🚨" if int(pred) == 1 else "Genuine Job ✅"
            if hasattr(model, "predict_proba"):
                prob_fake = float(model.predict_proba(features)[:, 1][0])

        except Exception as e:
            flash(f"Prediction failed: {e}", "error")

    return render_template("index.html",
                           prediction=prediction,
                           prob_fake=prob_fake)


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
