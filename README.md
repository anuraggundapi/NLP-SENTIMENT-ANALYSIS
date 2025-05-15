# NLP-SENTIMENT-ANALYSIS
## 📌 Project Overview

This project focuses on *Sentiment Analysis using Natural Language Processing (NLP)* techniques to analyze customer reviews collected from Amazon. Sentiment Analysis, a subfield of NLP, involves classifying text into subjective categories such as positive, negative, or neutral. It has become a critical tool for businesses to understand customer feedback, product perception, and overall satisfaction.

In this project, we preprocess raw textual data, explore linguistic patterns, and apply various machine learning algorithms to accurately predict the sentiment of a review. The final model is deployed using *Streamlit*, allowing users to interact with the application by entering custom reviews for real-time sentiment prediction.

---

## 🎯 Project Objective

The primary objectives of this project are:

1. *Perform Exploratory Data Analysis (EDA)* to understand sentiment distribution and word-level insights.
2. *Preprocess the Text Data* including cleaning, tokenization, lemmatization, and vectorization.
3. *Feature Engineering* with techniques like TF-IDF, and selection of impactful features.
4. *Build and Compare Machine Learning Models*:

   * Logistic Regression
   * Naive Bayes
   * Support Vector Machine (SVM)
   * Decision Tree
   * Random Forest
   * XGBoost
   * KNN
5. *Evaluate Model Performance* using metrics such as Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. *Select the Best-Performing Model* for deployment.
7. *Deploy the Model via Streamlit* for user-friendly interaction with the sentiment prediction tool.

## 🛠️ Tools & Technologies Used

* *Python 3.x*
* *Pandas, NumPy* – Data manipulation
* *Matplotlib, Seaborn* – Data visualization
* *NLTK, spaCy, re* – Text preprocessing
* *Scikit-learn* – Machine Learning models
* *XGBoost* – Gradient boosting model
* *Streamlit* – Web application deployment
* *Jupyter Notebook* – Development and experimentation

Model Accuracy:

1. Naive Bayes: 92
2. SVM: 93
3. Random Forest Model:94
4. KNN:91
5. XG Boost:94
6. Decision Tree:93
7. Logistic Regression:91

### ✅ *1. Project Pipeline Overview*

1. *Data Loading & Initial Exploration*
2. *Data Preprocessing & Cleaning*
3. *EDA & Visualization*
4. *Text Processing & Sentiment Analysis (TextBlob)*
5. *Outlier Detection (Isolation Forest)*
6. *Feature Engineering*

   * Encoding
   * PCA
   * Predictive Power Score (PPS)
7. *Model Building & Evaluation (one-by-one with GridSearchCV)*

   * Naive Bayes
   * Logistic Regression
   * Decision Tree
   * Random Forest
   * SVM
   * KNN
   * XGBoost
8. *Model Comparison*
9. *Model Deployment*

   * Export .pkl
   * Streamlit app.py

### 🚀 How to Run the App
Ensure the following files are in the same directory:

1. app.py – Streamlit application code
2. sentiment_model.pkl – Your trained ML model (e.g., XGBoost)
3. tfidf_vectorizer.pkl – Saved TF-IDF vectorizer
4. requirements.txt – Dependencies file

1. Create a virtual environment (recommended):

   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   

2. Install dependencies:

   bash
   pip install -r requirements.txt
   

3. Launch the app:

   bash
   streamlit run app.py

   
## 📈 Key Results

* Achieved up to *90% accuracy* with the best-performing model 
* Successfully deployed a sentiment prediction tool.
* Discovered frequent terms associated with positive and negative sentiments via EDA.
   
