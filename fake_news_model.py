import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data/news.csv")
print(df['label'].value_counts())
print(df['label'].unique())


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# Combine title and text
X = (df['title'] + " " + df['text']).apply(clean_text)
y = df['label']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vector, y_train)


# Predictions
y_pred = model.predict(X_test_vector)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Test with custom news
news = ["NASA confirms water on Mars"]

news_vector = vectorizer.transform(news)

prediction = model.predict(news_vector)

print("\nPrediction:", prediction)


# Show top words influencing prediction
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_fake_indices = coefficients.argsort()[:20]
top_real_indices = coefficients.argsort()[-20:]

print("\nTop words indicating FAKE news:")
for i in top_fake_indices:
    print(feature_names[i])

print("\nTop words indicating REAL news:")
for i in reversed(top_real_indices):
    print(feature_names[i])


# Save model
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel saved successfully!")