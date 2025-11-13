# Import required libraries
import pandas as pd
import numpy as np
import re
import pickle
import nltk
import tensorflow as tf
import random

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources for text processing and sentiment analysis
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize tools for preprocessing
stop_words = set(stopwords.words('english'))                  # English stop words list
lemmatizer = WordNetLemmatizer()                              # Lemmatizer for reducing words to base form
vader = SentimentIntensityAnalyzer()                          # Pretrained sentiment analyzer (VADER)


# Function to augment text data by replacing words with synonyms
def synonym_augment(text, n=2):
    words = text.split()
    new_words = words.copy()
    random.shuffle(new_words)                                # Shuffle to get random replacement
    num_replaced = 0

    for word in new_words:
        synonyms = wordnet.synsets(word)                      # Fetch synsets (groups of synonyms)
        if synonyms:
            lemmas = synonyms[0].lemma_names()                # Extract lemma names from first synset
            for syn in lemmas:
                if syn != word and syn.isalpha():             # Replace with a different alphabetical synonym
                    text = text.replace(word, syn.replace("_", " "), 1)
                    num_replaced += 1
                    break
        if num_replaced >= n:                                 # Stop after 'n' replacements
            break
    return text


# Function to clean and lemmatize text
def clean_text_advanced(text):
    text = text.lower()                                       # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)                   # Remove punctuation and non-letters
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # Remove stopwords and lemmatize
    return " ".join(words)


# Load and preprocess dataset
df = pd.read_csv("Lastt_training_data_combined_extended.csv", encoding="latin1")
df = df[["User_Text", "PHQ8_Binary"]].dropna()               # Keep only relevant columns and drop missing data
df["User_Text"] = df["User_Text"].astype(str).apply(clean_text_advanced)  # Clean and lemmatize user text
df["PHQ8_Binary"] = df["PHQ8_Binary"].astype(int)            # Ensure target variable is integer type

# ✨ Augment samples labeled as depressed (PHQ8_Binary == 1)
augmented_texts = []
augmented_labels = []

for i in range(len(df)):
    if df.iloc[i]['PHQ8_Binary'] == 1:
        original = df.iloc[i]['User_Text']
        new_text = synonym_augment(original, n=2)            # Apply synonym replacement
        augmented_texts.append(new_text)
        augmented_labels.append(1)                           # Keep same label (depressed)

# Create a new DataFrame with the augmented data
df_aug = pd.DataFrame({
    "User_Text": augmented_texts,
    "PHQ8_Binary": augmented_labels
})

# Combine original and augmented datasets
df = pd.concat([df, df_aug], ignore_index=True)
print(f"📈 Augmented {len(df_aug)} depressed samples")

# Add sentiment score from VADER as a new feature
df["Sentiment_Score"] = df["User_Text"].apply(lambda x: vader.polarity_scores(x)["compound"])

# Set the maximum number of unique words to keep (most frequent 10,000)
# Helps limit vocabulary size and reduce model complexity
vocab_size = 10000
# Set maximum length of each user input sequence (in tokens)
# Longer sequences will be truncated, shorter ones padded
# This ensures a consistent input shape for the model
max_length = 300
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(df["User_Text"])                      # Fit tokenizer on user text
sequences = tokenizer.texts_to_sequences(df["User_Text"])    # Convert texts to sequences of integers
X_text = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post")

# Combine tokenized text features with sentiment scores
X_sentiment = df["Sentiment_Score"].values.reshape(-1, 1)    # Reshape sentiment scores for concatenation
X_combined = np.concatenate([X_text, X_sentiment], axis=1)   # Final feature matrix
y = df["PHQ8_Binary"].values                                 # Target labels

# Save processed features and tokenizer
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

np.save("X_combined.npy", X_combined)
np.save("y.npy", y)

# Final logs and verification
print("✅ Preprocessing complete.")
print("🧠 X_combined shape:", X_combined.shape)
print("🔖 Label sample:", y[0])
print("🎯 Sample sentiment:", X_sentiment[0])
