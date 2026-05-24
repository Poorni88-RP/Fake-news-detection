"""
Veritas AI - Offline Naive Bayes NLP Classifier
This script implements a complete, error-free machine learning pipeline for fake news detection
using Natural Language Processing (NLP) text preprocessing and a Multinomial Naive Bayes model.

To run this file locally, install the required packages:
    pip install pandas numpy scikit-learn nltk
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download necessary NLTK data resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"NLTK Download warning (safe to ignore if already offline): {e}")

class FakeNewsNaiveBayesPipeline:
    def __init__(self):
        # We limit features to 5000 to prevent overfitting and keep vocabulary dimensions organized
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = MultinomialNB(alpha=1.0)
        self.stemmer = PorterStemmer()
        
        # Safe fallback for stop words if nltk fails to load
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"the", "a", "an", "and", "but", "if", "or", "because", "as", "what", "which", "this", "that"}

    def clean_text(self, text):
        """Processes raw text by removing punctuation, converting to lowercase, and stemming."""
        if not isinstance(text, str):
            return ""
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()
        words = text.split()
        
        # Filter stopwords and apply stemming
        cleaned_words = [
            self.stemmer.stem(word) for word in words 
            if word not in self.stop_words
        ]
        return " ".join(cleaned_words)

    def train(self, df, text_column='text', label_column='label'):
        """
        Trains the TF-IDF and Naive Bayes model.
        df: Pandas DataFrame
        text_column: Column name containing the news body
        label_column: Column name containing label (1 for Real, 0 for Fake)
        """
        print("Cleaning text data using standard NLP Stemmers...")
        df['processed_text'] = df[text_column].apply(self.clean_text)
        
        X = df['processed_text']
        y = df[label_column]
        
        # Train-Test Split (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Extracting TF-IDF features (vocabulary limit = 5000 tokens)...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Fitting Multinomial Naive Bayes model...")
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate performance
        predictions = self.model.predict(X_test_vec)
        acc = accuracy_score(y_test, predictions)
        print("\n=== Offline Model Evaluation Metrics ===")
        print(f"Total Test Accuracy Score: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Fake', 'Real']))
        
        return acc

    def predict_credibility(self, raw_text):
        """Returns predicted class label and real-time probability likelihoods."""
        cleaned = self.clean_text(raw_text)
        vectorized = self.vectorizer.transform([cleaned])
        
        # Get conditional probability array
        proba = self.model.predict_proba(vectorized)[0] # [P(Fake), P(Real)]
        p_fake = float(proba[0])
        p_real = float(proba[1])
        
        label = "REAL" if p_real > p_fake else "FAKE"
        credibility_score = p_real * 100
        
        return {
            "prediction": label,
            "real_probability": p_real,
            "fake_probability": p_fake,
            "credibility_percent": round(credibility_score, 2)
        }

if __name__ == "__main__":
    print("==================================================================")
    print("  VERITAS AI - Standalone Naive Bayes Classifier Engine   ")
    print("==================================================================")
    
    # Generate balanced sample data for demonstration (10 samples to support clean train-test split stratify)
    mock_data = {
        'text': [
            "Astronomers made a breakdown observing water vapor in protoplanetary disk around a young star.",
            "SHOCKING miracle flower GlycoCure instantly cures diabetes in 48 hours but government hides report!",
            "Central bankers announced moderate interest rate cuts following stabilized CPI index reports.",
            "BREAKING: Underground whistleblower reveals alien mind frequency towers built in secret mountains!",
            "Major medical journal published clinical trials validating new monoclonal antibody treatments.",
            "Urgent leak: Secret space force bases located inside hollow Moon are tracking your thoughts!",
            "Global trade ministers finalized high-level tariff reductions on organic agricultural produce.",
            "Miracle water source in Andes reportedly reverses aging process within seven days of drinking.",
            "Municipal authorities announced extensive funding plans to restore regional forestry ecosystems.",
            "Conspiracy exposes standard electric vehicles having remote control switches to trap citizens!"
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 1: Real News, 0: Fake News
    }
    df = pd.DataFrame(mock_data)
    
    pipeline = FakeNewsNaiveBayesPipeline()
    pipeline.train(df, 'text', 'label')
    
    # Run prediction test 1
    test_article_1 = "Severe healthcare research confirms that raw lemon juice acts as high performance immunity booster."
    result_1 = pipeline.predict_credibility(test_article_1)
    
    # Run prediction test 2
    test_article_2 = "Astronomers confirm presence of rocky core structures inside distant celestial bodies."
    result_2 = pipeline.predict_credibility(test_article_2)

    print("\n=== Live Sample Inference Outputs ===")
    print(f"Test 1 Input: '{test_article_1}'")
    print(f"Test 1 Verdict: {result_1['prediction']} ({result_1['credibility_percent']}% Real Likelihood)")
    print("-" * 50)
    print(f"Test 2 Input: '{test_article_2}'")
    print(f"Test 2 Verdict: {result_2['prediction']} ({result_2['credibility_percent']}% Real Likelihood)")
    print("==================================================================")
