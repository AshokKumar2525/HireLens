# HireLens 🧠🔍
HireLens – An AI-powered web application that analyzes job postings and flags potential fraudulent ads using NLP and machine learning.

## 📌 Overview
HireLens is a machine learning–driven web application that helps users identify **potentially fraudulent job postings**.  
It uses **Natural Language Processing (NLP)** and a **Random Forest Classifier** trained on 18,000+ job postings (with 800+ fake jobs) to classify whether a job post is **real or fake**.  

The project is built with **Flask** for deployment, allowing users to paste job descriptions into a web form and instantly check their authenticity.

---

## ✨ Features
- 🔎 **Paste & Check**: Users can submit job descriptions via a simple UI.  
- 🧹 **NLP Preprocessing**: Lemmatization, stopword removal, and cleaning.  
- 🌲 **Random Forest Classifier** with ~96% accuracy.  
- 📊 **Data Insights & Visualizations** (EDA included in notebook).  
- 🌐 **Flask Web App** for real-time predictions.  

---

## 🛠 Tech Stack
- **Python** 🐍  
- **Scikit-learn, Pandas, NumPy, NLTK**  
- **Flask** (backend & UI deployment)  
- **Matplotlib / Seaborn** for visualizations  
- **Joblib** for model persistence  

---

## 📂 Project Structure
HireLens/
│── app.py                  # Flask app 
│── model.joblib            # Trained ML model (Random Forest)
│── requirements.txt        # Project dependencies
│── README.md               
│
├── notebook/               
│   └── fake_job_detector.ipynb
│
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│       └── logo.png
│
└── templates/              
    ├── index.html          # Homepage (form to paste job post)
    ├── result.html         # Prediction result page
    └── about.html          # Optional: About project


---

## ⚡️ Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/HireLens.git
   cd HireLens
   ```

2.  **Install Dependencies**
    ```bash
     pip install -r requirements.txt
    ```
3. Run the Flask App
    ```bash
      python app.py
    ```
## 📊 Dataset
- **Source:** Kaggle – Fake Job Posting Prediction
- **Collected By:** University of the Aegean
- ~18,000 job postings (800+ fake jobs)

## 🚀 Future Scope
- 🌐 Extend to LinkedIn/Indeed scraping for live job detection.
- 📱 Build a React-based frontend for better UI.
- ⚡️ Try advanced models (XGBoost, LSTM, Transformers).
