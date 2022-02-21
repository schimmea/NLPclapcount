import pandas as pd
import numpy as np
import multiprocessing as mp
import time
# Library re provides regular expressions functionality
import re

# Saving and loaded objects
import pickle

# Library beatifulsoup4 handles html
from bs4 import BeautifulSoup

# Standard NLP workflow
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


def get_wordnet_pos(word):
    """Map POS tag to first character for lemmatization"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def clean_articles(text):
    lemmatizer = WordNetLemmatizer()
     # remove newlines
    review_text = text.replace("\n", " ")
    review_text = review_text.replace("\xa0", " ")
    # remove html content
    review_text = BeautifulSoup(review_text, "html.parser").get_text()
    # remove URLs
    review_text = re.sub(r'http\S+', '', review_text)
    review_text = re.sub(r'www\.\S+', '', review_text)
    # remove bibliographies (can be at start or end):
    # assuming bibliography is shorter than text:
    test_split = str(review_text).split("Bibliography")
    review_text = test_split[np.where(np.array([len(x) for x in test_split]) == max([len(x) for x in test_split]))[0].tolist().pop()]
    # remove non-alphabetic characters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # fix contractions
    review_text = re.sub(r"['’]ll", " will", review_text)
    review_text = re.sub(r"['’]ve", " have", review_text)
    review_text = re.sub(r"['’]re", " are", review_text)
    review_text = re.sub(r"['’]d", " would", review_text)
    review_text = re.sub(r"['’]m", " am", review_text)
    review_text = re.sub(r"(?i)there['’]s", "there is", review_text)
    review_text = re.sub(r"(?i)that['’]s", "that is", review_text)
    review_text = re.sub(r"(?i)it['’]s", "it is", review_text)
    review_text = re.sub(r"(?i)he['’]s", "he is", review_text)
    review_text = re.sub(r"can['’]t", "cannot", review_text)
    review_text = re.sub(r"won['’]t", "will not", review_text)
    review_text = re.sub(r"n['’]t", " not", review_text)
    review_text = review_text.replace("gonna", "going to")
    review_text = review_text.replace("wanna", "want to")
    review_text = review_text.replace("gotta", "got to")
    # tokenize the sentences
    words = word_tokenize(review_text.lower())
    # filter stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatize each word to its lemma
    lemma_words = [lemmatizer.lemmatize(i, get_wordnet_pos(i)) for i in words]
    return lemma_words

def mp_clean_helper(df):
    print('*' * 40)
    print(f'{time.strftime("%H:%M:%S", time.localtime())} Worker {mp.current_process().name.lstrip("SpawnPoolWorker-")} : Cleaning {df.shape[0]} articles.')
    res = df.apply(clean_articles)
    detokenized = [TreebankWordDetokenizer().detokenize(article) for article in res]
    out_df = pd.Series(detokenized, index=df.index, dtype="str")
    print(f'{time.strftime("%H:%M:%S", time.localtime())} Worker {mp.current_process().name.lstrip("SpawnPoolWorker-")} : DONE')
    print('*' * 40)
    return out_df


if __name__ == '__main__':
    with open('train_nodupl.pkl', 'rb') as file_name:
        train_title = pickle.load(file_name).title.astype("str")

    with open('train_nodupl.pkl', 'rb') as file_name:
        train_text = pickle.load(file_name).text.astype("str")

    test_title = pd.read_csv("Data/test.csv", sep=",", encoding="utf-8", index_col="index").Header.astype("str")
    test_text = pd.read_csv("Data/test.csv", sep=",", encoding="utf-8", index_col="index").Text.astype("str")

    no_parts = mp.cpu_count() - 1

    train_title = np.array_split(train_title, no_parts)
    train_text = np.array_split(train_text, no_parts)
    test_title = np.array_split(test_title, no_parts)
    test_text = np.array_split(test_text, no_parts)

    with mp.Pool(no_parts) as p:
        train_title_out = p.map(mp_clean_helper, train_title)
        train_text_out = p.map(mp_clean_helper, train_text)
        test_title_out = p.map(mp_clean_helper, test_title)
        test_text_out = p.map(mp_clean_helper, test_text)

    train_title_clean = pd.concat(train_title_out)
    train_text_clean = pd.concat(train_text_out)
    test_title_clean = pd.concat(test_title_out)
    test_text_clean = pd.concat(test_text_out)

    with open('train_titles_clean.pkl', 'wb') as file_name:
        pickle.dump(train_title_clean, file_name)
    with open('train_text_clean.pkl', 'wb') as file_name:
        pickle.dump(train_text_clean, file_name)
    with open('test_titles_clean.pkl', 'wb') as file_name:
        pickle.dump(test_title_clean, file_name)
    with open('test_text_clean.pkl', 'wb') as file_name:
        pickle.dump(test_text_clean, file_name)
