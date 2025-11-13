# preprocessing_utils.py
import re
import random
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text_advanced(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def synonym_augment(text, n=2):
    words = text.split()
    new_words = words.copy()
    random.shuffle(new_words)
    num_replaced = 0
    for word in new_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            lemmas = synonyms[0].lemma_names()
            for syn in lemmas:
                if syn != word and syn.isalpha():
                    text = text.replace(word, syn.replace("_", " "), 1)
                    num_replaced += 1
                    break
        if num_replaced >= n:
            break
    return text
