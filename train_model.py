import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("Loading dataset...")

# ✅ CORRECT DATASET PATH
data = pd.read_csv(
    "Dataset/spam.csv",
    encoding="latin-1"
)

# Clean dataset
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
data = data.dropna()

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data['text'] = data['text'].str.lower()

# Features and labels
X = data['text']
y = data['label']

# Train model using Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

pipeline.fit(X, y)

# Save trained model
with open("spam_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Training complete. spam_pipeline.pkl saved.")
