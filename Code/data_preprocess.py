import os 
import string
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm


import spacy
from spacy.lang.en.stop_words import STOP_WORDS
# !python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')

# ### Tokenize, lemmatize and remove stopping words
def tokenize(sentence,method='spacy'):
# Tokenize and lemmatize text, remove stopwords and punctuation

    punctuations = string.punctuation
    stopwords = list(STOP_WORDS)

    if method=='nltk':
        # Tokenize
        tokens = nltk.word_tokenize(sentence,preserve_line=True)
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        # Lemmatize
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        tokens = " ".join([i for i in tokens])
    else:
        # Tokenize
        with nlp.select_pipes(enable=['tokenizer','lemmatizer']):
            tokens = nlp(sentence)
        # Lemmatize
        tokens = [word.lemma_.lower().strip() for word in tokens]
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stopwords and word not in punctuations]
        tokens = " ".join([i for i in tokens])
    return tokens

def load_data():
    #load data and select useful columns
    print("Loading data...")
    cols = ['ProductId', 'UserId', 'Score', 'Text']
    select_data = pd.read_csv(os.path.join(DATA_PATH, 'Reviews.csv'))[cols]
    print("Load data done!")
    return select_data

def process_data(data):
    print("Processing data...")
    #split the data
    train_data, test_data = train_test_split(data, random_state=0, test_size=0.3)
    
    # ### Label Sentiment
    train_data = label_sentiment(train_data)
    test_data = label_sentiment(test_data)

    # Process the training set text
    print("Processing the training set text...")
    tqdm.pandas()
    train_data['processed_text'] = train_data['Text'].progress_apply(lambda x: tokenize(x,method='nltk'))

    # Process the test set text
    print("Processing the test set text...")
    tqdm.pandas()
    test_data['processed_text'] = test_data['Text'].progress_apply(lambda x: tokenize(x,method='nltk'))

    print("Create bins...")
    #compute the average sentiment for each user and product
    user_sent = train_data.groupby(['UserId'])['sentiment'].agg(['count', 'mean']).reset_index()
    prod_sent = train_data.groupby(['ProductId'])['sentiment'].agg(['count', 'mean']).reset_index()

    #create bins for sentiment
    user_sent_bin = create_bins(user_sent)[['UserId', 'bin_sent']]
    user_sent_bin.columns = ["UserId", "user_bin_sent"]
    prod_sent_bin = create_bins(prod_sent)[['ProductId', 'bin_sent']]
    prod_sent_bin.columns = ['ProductId', 'prod_bin_sent']

    #create new columns for bined sentiment in dataset
    train_data = train_data.merge(user_sent_bin, how = "inner", left_on="UserId", right_on="UserId")
    train_data = train_data.merge(prod_sent_bin, how = "inner", left_on="ProductId", right_on="ProductId")
    test_data = test_data.merge(user_sent_bin, how = "inner", left_on="UserId", right_on="UserId")
    test_data = test_data.merge(prod_sent_bin, how = "inner", left_on="ProductId", right_on="ProductId")

    print("Bins created!")
    print("Process done!")
    return train_data, test_data

def label_sentiment(data):
    data.loc[data['Score'] == 5, ["sentiment"]] = 1
    data.loc[data['Score'] == 4, ["sentiment"]] = 0
    data.loc[data['Score'] <= 3, ["sentiment"]] = -1
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', type = str, default=None)
    args = parser.parse_args()
    return args

def create_bins(data):
    #group the sentiment into bins according to its quantile
    n = 20
    q = [i/n for i in range(1, n)]
    quant = data['mean'].quantile(q)

    #create bins
    bins = list(set(quant))
    bins.sort(reverse=True)

    #map bins with sentiment
    data['bin_sent'] = len(bins)
    for i in range(len(bins)-1):
        idx = (data['mean'] < bins[i]) & (data['mean'] >= bins[i+1])
        data.loc[idx, ['bin_sent']] = len(bins) - i - 1
    
    return data

def main():
    # ##Load Data
    data = load_data()

    # ## Preprocess Data
    train_data, test_data = process_data(data)

    print("Save data...")
    train_data.to_csv(os.path.join(DATA_PATH, "train.csv"), index = False)
    test_data.to_csv(os.path.join(DATA_PATH, "test.csv"), index = False)

if __name__ == "__main__":
    arg = parse_args()
    if arg.code_path == None:
        CODE_PATH = os.getcwd()
    else:
        CODE_PATH = arg.code_path
    BASE_PATH = os.path.abspath(os.path.join(CODE_PATH, ".."))
    DATA_PATH = os.path.join(BASE_PATH, "Data")
    MODEL_PATH = os.path.join(BASE_PATH, "Model")
    os.chdir(BASE_PATH)
    
    main()