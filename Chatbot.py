import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("chat_sentiment.csv") 

# Preprocess text
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['Chat'] = data['Chat'].apply(preprocess_text)

# Split data into training and test sets
X = data['Chat']
y = data['sentiment']

# Convert text to numerical features
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy: ", accuracy_score(y_test, y_pred))

# Function to predict the sentiment of a new chat input
def predict_sentiment(new_chat):
    # Preprocess the new chat input
    new_chat_processed = preprocess_text(new_chat)
    
    # processed chat
    new_chat_tfidf = tfidf.transform([new_chat_processed])
    
    # sentiment prediction
    sentiment = model.predict(new_chat_tfidf)[0]
    
    return sentiment

# chat input from the user
new_chat_input = input("Enter your message: ")
predicted_sentiment = predict_sentiment(new_chat_input)
print(f"The predicted sentiment of the chat is: {predicted_sentiment}")