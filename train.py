import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import spacy

# Load a pre-trained spaCy model with GloVe embeddings
nlp = spacy.load("en_core_web_sm")

# Load the GloVe embeddings
glove_path = "glove_embeddings/glove.6B.100d.txt"
glove_embeddings = {}
with open(glove_path, encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype="float32")
        glove_embeddings[word] = vector

# Step 1: Load the dataset
data = pd.read_csv("data/preprocessed_data.csv")
# Labeling
sentiment_column = 'SentimentScore'
data['Label'] = data[sentiment_column].apply(
    lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

# Step 2: Tokenize the Tweets using the loaded GloVe embeddings
def vectorize_with_glove(text):
    doc = nlp(text)
    token_vectors = [token.vector for token in doc if not token.is_stop]
    return np.mean(token_vectors, axis=0) if token_vectors else np.zeros(300)  # Assuming 300 dimensions

data['tokenized_tweet'] = data['PreprocessedTweet'].apply(vectorize_with_glove)

# Step 3: Split data into training and testing sets
X = data['tokenized_tweet']
y = data['Label']  # Label Column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest model for sentiment prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(list(X_train), y_train)

# Step 5: Evaluate the model
y_pred = model.predict(list(X_test))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", confusion)

# Save the trained model
model_path = "model/sentiment_model_rf.pkl"
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")
