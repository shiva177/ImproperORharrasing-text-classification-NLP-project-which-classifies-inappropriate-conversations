import string
from num2words import num2words
from nltk.stem import WordNetLemmatizer
import pandas as pd
import pickle
import json

PREDICTION_TO_LABEL = prediction_to_label = {0: "Negative", 1: "Positive"}

with open("models/pipeline.pickle", "rb") as f:
    ml_pipeline = pickle.load(f)

with open("util/abbreviations_english.json") as f:
    abbreviations = json.load(f)


def numbers_to_words(text: str) -> str:
    """
    Convert numbers written as strings to alphabetic characters

    Parameter
    ---------
    text: str
        Input text to process

    Returns
    -------
    str
        Cleaned text
    """
    cleaned_text = ""

    for word in text.split(" "):
        if word.isnumeric() and int(word) < 1000000000:
            cleaned_text = f"{cleaned_text} {num2words(word)}"
        else:
            cleaned_text = f"{cleaned_text} {word}"
    return cleaned_text


def preprocess_text(text, abbreviations) -> str:
    """
    Clean and extract important text from input text

    Parameters
    ----------
    text: str
        Input text to classify

    abbreviations: dict
        English common abbreviations to replace with original words

    Returns
    -------
    str
        Cleaned text
    """

    # Delete punctuation
    text.translate(str.maketrans("", "", string.punctuation))

    # Replace double space with only one
    text = " ".join([word.lower() for word in text.split(" ") if word])

    # Convert numbers to words
    text = numbers_to_words(text)

    # Replace abbreviations
    text = " ".join(
        [
            abbreviations[word] if word in abbreviations.keys() else word
            for word in text.split(" ")
        ]
    )

    # Delete punctuation
    text.translate(str.maketrans("", "", string.punctuation))

    # Delete alphanumeric characters
    text = " ".join([word for word in text.split(" ") if word.isalpha()])

    # Lemmatize text
    wordnet_lemmatizer = WordNetLemmatizer()
    text = " ".join(
        [
            wordnet_lemmatizer.lemmatize(word, pos="v")
            for word in text.split(" ")
        ]
    )

    return text


def ml_predict(text):
    preprocessed_text = preprocess_text(text, abbreviations)

    # Predict input text
    prediction = ml_pipeline.predict(pd.Series(preprocessed_text))

    return PREDICTION_TO_LABEL[prediction[0]]
