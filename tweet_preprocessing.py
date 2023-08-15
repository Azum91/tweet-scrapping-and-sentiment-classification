import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from textblob import TextBlob
import spacy

# Spell correctiong function
def correct_spelling(tweet):
    blob = TextBlob(tweet)
    corrected_tweet = blob.correct()
    return str(corrected_tweet)

# Sentiment Score fetching Higher score = to positive.
def get_sentiment_score(tweet):
    blob = TextBlob(tweet)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Text preprocessing functions
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove user mentions
    text = re.sub('[^A-Za-z0-9]+', ' ', text)  # Remove non-alphanumeric characters
    text = re.sub(r'\w+\s+\d+', '', text) # Remove Date added from the text
    text_words = text.split()
    text = ' '.join(word for word in text_words if word.isalpha())
    return text

def preprocess_tweet(tweet,nlp):

    tweet = clean_text(tweet)
    tweet = nlp(tweet)
    filtered_tokens = [token.text for token in tweet if token.pos_ != 'NOUN']
    tweet = ' '.join(filtered_tokens)
    words = word_tokenize(tweet.lower())  # Tokenization and lowercasing
    words = [word for word in words if word not in stopwords.words('english')]  # Stopword removal
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # Stemming
    tweet = ' '.join(words)
    tweet = correct_spelling(tweet) #words correction
    return tweet

# Apply preprocessing to the "TweetText" column

def convert_to_int(string_with_suffix):
    if pd.isna(string_with_suffix):
        return 0
    elif isinstance(string_with_suffix, str):
        if string_with_suffix[-1] == 'K':
            print(string_with_suffix)
            return int(float(string_with_suffix[:-1]) * 1000)
        else:
            return int(string_with_suffix.replace(',', ''))
    else:
        print(string_with_suffix)
        return int(string_with_suffix)

if __name__ == "__main__":

    # Download NLTK resources (only need to run this once)
    nltk.download('punkt')
    nltk.download('stopwords')

    # Loading NLP libary for Nouns
    nlp = spacy.load("en_core_web_sm")

    # Load the CSV file
    csv_file = './data/Quran Respect Emotion Data.csv'  # Replace with the path to your CSV file
    data = pd.read_csv(csv_file)
    data['PreprocessedTweet'] = data['TweetText'].apply(preprocess_tweet, args=(nlp,))
    # Converting Numbers Columns to Integer Type
    data['LikeCount'] = data['LikeCount'].apply(convert_to_int)
    data['ReplyCount'] = data['ReplyCount'].apply(convert_to_int)
    data['RetweetCount'] = data['RetweetCount'].apply(convert_to_int)
    # SentimentScore Predictor
    data['SentimentScore'] = data['PreprocessedTweet'].apply(get_sentiment_score)
    # Sample of the data
    print(data.head())

    # Save the preprocessed data to a new CSV file
    preprocessed_csv_file = './data/preprocessed_data.csv'
    data.to_csv(preprocessed_csv_file, index=False)

    print("Preprocessing completed and saved to", preprocessed_csv_file)