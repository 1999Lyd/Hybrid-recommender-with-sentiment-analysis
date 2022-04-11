import os 
import pickle
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

def load_data():
    print("Loading data...")
    train_data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print("Load data done!")
    return train_data, test_data

# ## Create features using word counts
def build_features(train_data, test_data, ngram_range, method='count'):
    print("Building features...")
    if method == 'tfidf':
        # Create features using TFIDF
        vec = TfidfVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_data['processed_text'])
        X_test = vec.transform(test_data['processed_text'])

    else:
        # Create features using word counts
        vec = CountVectorizer(ngram_range=ngram_range)
        X_train = vec.fit_transform(train_data['processed_text'])
        X_test = vec.transform(test_data['processed_text'])
    print("Build features done!")
    return X_train, X_test, vec

def save_model(nlp, model):
    print("Saving models...")
    with open(os.path.join(MODEL_PATH, "tfidf.pickle"), 'wb') as f:
        pickle.dump(nlp, f)

    with open(os.path.join(MODEL_PATH, "tfidf_reg.pickle"), 'wb') as f:
        pickle.dump(model, f)
    print("Model saved!")

def main():
    # ##Load Data
    train_data, test_data = load_data()

    # Create features
    method = 'tfidf'
    ngram_range = (1, 2)
    X_train, X_test, nlp = build_features(train_data, test_data, ngram_range, method)

    # ## Train the model
    # Train a classification model using logistic regression classifier
    print("Training models")
    y_train = train_data['sentiment']
    logreg_model = LogisticRegression(solver='saga')
    logreg_model.fit(X_train,y_train)
    print("Train models done!")

    preds = logreg_model.predict(X_train)
    acc = sum(preds==y_train)/len(y_train)
    print('Accuracy on the training set is {:.3f}'.format(acc))

    # ## Evaluate model

    # Evaluate accuracy on the test set
    print("Evaluating models...")
    y_test = test_data['sentiment']
    test_preds = logreg_model.predict(X_test)
    print("Evaluation done!")

    test_acc = sum(test_preds==y_test)/len(y_test)
    print('Accuracy on the test set is {:.3f}'.format(test_acc))

    #save model
    save_model(nlp, logreg_model)

    #save predict sentiment
    print("Saving data...")
    train_data['sentiment_pred'] = logreg_model.predict(X_train)
    test_data['sentiment_pred'] = logreg_model.predict(X_test)

    train_data.to_csv(os.path.join(DATA_PATH, "tfidf_train_bins.csv"), index = False)
    test_data.to_csv(os.path.join(DATA_PATH, "tfidf_test_bins.csv"), index = False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_path', type = str, default=None)
    args = parser.parse_args()
    return args

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

