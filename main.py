import shutil
from textblob import TextBlob
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import os

# Define the relative path to your dataset file
file_name = 'ds.csv'
file_nameZ = 'ds2.csv'

# Get the current working directory (where your script is located)
current_directory = os.getcwd()

# Construct the absolute path to your dataset file using os.path.join
file_path = os.path.join(current_directory, file_name)
file_pathZ = os.path.join(current_directory, file_nameZ)

df = pd.read_csv(file_path)

# Split the dataset into training and testing sets
X = df['phrase']  # Your text data
y = df['sentiment']  # Your sentiment labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Choose a machine learning model (e.g., LinearSVC)
model = LinearSVC(dual=True)

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results

# Download the necessary resources for TextBlob (if not already downloaded)
nltk.download('punkt')

# Function to analyze sentiment


def analyze_sentiment(user_input):
    analysis = TextBlob(user_input)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0:
        return "positive"
    elif sentiment_score == 0:
        return "neutral"
    else:
        return "negative"

# Function to clear and copy data


def clear_and_copy_data():
    # Clear the data in ds.csv
    with open(file_path, 'w') as file:
        file.write('phrase,sentiment\n')

    # Copy data from ds2.csv to ds.csv
    shutil.copy(file_pathZ, file_path)


# Chatbot loop
print("Chatbot: Hello! I can analyze the sentiment of your input. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Thank You So Much!, Have a good day")
        print("Sentiment Analysis Chatbot -- Made By Jayesh Chak (thewabisabiway.learnjc@gmail.com)")
        clear_and_copy_data()
        break

    sentiment = analyze_sentiment(user_input)
    print(f"Chatbot: The sentiment of your input is {sentiment}.")
    # Create a new DataFrame for the new data
    new_dataZ = {'phrase': [user_input], 'sentiment': [sentiment]}
    new_df = pd.DataFrame(new_dataZ)

    # Append the new data to the existing dataset
    df = pd.concat([df, new_df], ignore_index=True)

    # Save the updated dataset back to the file without overwriting
    df.to_csv(file_pathZ, index=False)
    
    
