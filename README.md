COMPANY:CODETECH IT SOLUTIONS NAME:Aditi Kodande INTERN ID::CT04DA652 DOMAIN:Machine learning BATCH DURATION:April 10th, 2025 to May 10th, 2025. MENTOR NAME:NEELA SANTOSH


---

# Sentiment Analysis on Text Reviews 💬🔍

This project builds a **Logistic Regression** model for **sentiment analysis** (positive or negative) using a **TF-IDF vectorizer** on a dataset of text reviews.

---

## 📚 Project Overview

The steps covered in this project:

- **Import Required Libraries**
- **Load Dataset** (with custom manual fixes if necessary)
- **Preprocess Text Data** (cleaning, lowercasing, removing URLs, HTML tags, etc.)
- **Encode Sentiment Labels** (`Positive` → 1, `Negative` → 0)
- **Train-Test Split**
- **TF-IDF Feature Extraction**
- **Model Building and Training** (Logistic Regression)
- **Evaluation** (accuracy, classification report, confusion matrix)
- **Model Saving** (using `joblib`)

---

## 📦 Requirements

Install the necessary Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 🚀 How to Run

1. **Clone** this repository or download the project files.
2. **Install** the required packages.
3. **Run the Python script** (or Jupyter Notebook).
4. **Check outputs** like accuracy score, classification report, and confusion matrix.
5. **Saved Models:**  
   After running, two files will be created:
   - `sentiment_model.pkl`
   - `tfidf_vectorizer.pkl`

These files can be used for deploying the model without retraining.

---

## 🛠 Project Details

- **Dataset:** `sentiment-analysis.csv`
- **Features:**
  - Text reviews (column: `Text`)
- **Target:**
  - Sentiment (Positive / Negative)

### Text Preprocessing:
- Lowercasing
- Removing URLs
- Removing HTML tags
- Removing non-alphabetic characters
- Removing extra spaces

### Machine Learning Model:
- **Model:** Logistic Regression
- **Feature Extraction:** TF-IDF Vectorizer (`max_features=5000`)
- **Train/Test Split:** 80% Training, 20% Testing

---

## 📈 Results

- **Accuracy Score** is printed after model evaluation.
- **Classification Report** provides precision, recall, and F1-score.
- **Confusion Matrix** visualizes true vs predicted classes.

Example outputs:
- Accuracy ~ 85%–95% (depending on dataset quality)
- Well-balanced precision and recall for both classes.

---

## 📊 Visualization

- **Confusion Matrix Heatmap** is generated for visual inspection of model performance.

---

## 🧠 Notes

- **Handling CSV Issues:**  
  If the CSV file has formatting issues (extra quotes, wrong commas), manual parsing and cleaning is performed.
- **Handling Missing Values:**  
  Rows with missing sentiment labels are dropped.
- **Saving the Model:**  
  Both the trained model and the TF-IDF vectorizer are saved for later use.

---

## 🔥 Future Work

- Try using advanced models like SVM, Random Forest, or Deep Learning (LSTM).
- Perform hyperparameter tuning using Grid Search.
- Apply more sophisticated text preprocessing (like stemming/lemmatization).
- Deploy the model using Flask, Django, or Streamlit.

---

## 🏷️ Example Saved Files

| File Name            | Description                      |
|----------------------|----------------------------------|
| sentiment_model.pkl   | Trained Logistic Regression model |
| tfidf_vectorizer.pkl  | Fitted TF-IDF Vectorizer           |

---

## 💬 Acknowledgements

- Thanks to Scikit-learn, Pandas, and Matplotlib teams for amazing libraries.
- Dataset used for educational purposes.

---
https://github.com/user-attachments/assets/ca89ac7a-398c-43d5-be75-c8a6d790f191
