🧠 MacBook Review Analysis using NLP
📘 Project Overview

This project performs sentiment analysis and text classification on MacBook product reviews using Natural Language Processing (NLP) techniques. The goal is to understand customer opinions, identify key sentiment trends, and visualize insights from real-world review data.

⚙️ Key Features

🧹 Data Cleaning: Removed URLs, numbers, punctuation, and stopwords.

🔤 Text Preprocessing: Tokenization, lemmatization using spaCy.

💬 Sentiment Scoring: Used TextBlob to calculate polarity scores.

📊 Feature Extraction: Converted text into TF-IDF vectors for machine learning models.

🧮 Model Training: Implemented multiple classifiers —

Logistic Regression (Accuracy: ~97%)

Multinomial Naive Bayes (Accuracy: ~97%)

Compared results with an Artificial Neural Network (ANN).

📈 Evaluation: Measured precision, recall, F1-score, and accuracy.

🔍 Insights: Identified most frequent positive/negative words and sentiment distribution among MacBook reviews.

🧰 Tech Stack

Python

Pandas, NumPy, Matplotlib, Seaborn

spaCy, TextBlob

scikit-learn

TensorFlow / Keras

NLTK

📑 Workflow

Data Collection – Imported MacBook reviews dataset (CSV).

Text Preprocessing – Cleaned and normalized textual data.

Exploratory Data Analysis (EDA) – Visualized review lengths, sentiment polarity, and word clouds.

Model Building – Trained multiple ML models for sentiment classification.

Evaluation & Comparison – Compared performance of Logistic Regression, Naive Bayes, and ANN.

Visualization – Displayed sentiment distribution and model accuracy comparisons.

🏁 Results

Best Model: Logistic Regression

Accuracy: 97.2%

Conclusion: Most reviews are positive, showing high customer satisfaction with MacBook performance, design, and battery life, though some negative feedback relates to pricing and overheating issues.

🚀 Future Enhancements

Integrate BERT / Transformer models for deeper context understanding.

Build a dashboard (Streamlit) to visualize sentiment interactively.

Add aspect-based sentiment analysis (battery, performance, price, etc.)."# Macbook-Review-Analysis" 
