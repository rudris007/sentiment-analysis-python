import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
data = pd.read_csv("data/reviews.csv")

# Preprocessing
X = data['review']
y = data['sentiment']

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Test custom input
while True:
    user_input = input("\nEnter a review (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)
    print(f"Sentiment: {prediction[0]}")
