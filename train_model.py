# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('fake_job_postings (1).csv')

# Combine text columns into a single column
df['text'] = df['description'] + ' ' + df['company_profile'] + ' ' + df['requirements'] + ' ' + df['benefits']

# Fit the TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=100)
X_tfidf = tfidf.fit_transform(df['text'])

# Train the Random Forest model
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_tfidf, df['fraudulent'])

# Save the TF-IDF vectorizer and the trained model
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(rf_clf, 'random_forest_model.pkl')

print("Model and vectorizer saved successfully.")